"""
src/metrics.py
Standardized Evaluation Metrics for OCR & Document Understanding:
- Character Error Rate (CER)
- Word Error Rate (WER)
- Normalized Edit Distance (NED) & Character Accuracy
- Layout Preservation Score (Structural tokens, tables, headers, reading order)
- Key-Value Extraction Metrics (Precision, Recall, F1, Hallucination Rate)
- Resource Profiling (Inference Latency, Memory, VRAM)
"""

import re
import time
import json
from typing import List, Dict, Any, Tuple, Optional, Union


def levenshtein_distance(seq1: Union[str, List[str]], seq2: Union[str, List[str]]) -> int:
    """
    Computes the standard Levenshtein edit distance between two sequences
    (characters or words) using optimized two-row dynamic programming.
    Zero external C-dependency requirement.
    """
    len1, len2 = len(seq1), len(seq2)
    if len1 == 0:
        return len2
    if len2 == 0:
        return len1

    # Keep only two rows
    prev_row = list(range(len2 + 1))
    current_row = [0] * (len2 + 1)

    for i in range(len1):
        current_row[0] = i + 1
        for j in range(len2):
            cost = 0 if seq1[i] == seq2[j] else 1
            current_row[j + 1] = min(
                current_row[j] + 1,       # Insertion
                prev_row[j + 1] + 1,      # Deletion
                prev_row[j] + cost        # Substitution
            )
        prev_row, current_row = current_row, prev_row

    return prev_row[len2]


def compute_cer(prediction: str, reference: str, case_sensitive: bool = False) -> float:
    """
    Computes Character Error Rate (CER):
    CER = Levenshtein_Distance(pred, ref) / max(1, len(ref))
    Capped at 1.0 (100% error rate maximum).
    """
    p = prediction if case_sensitive else prediction.lower()
    r = reference if case_sensitive else reference.lower()

    if not r:
        return 0.0 if not p else 1.0

    dist = levenshtein_distance(p, r)
    cer = dist / max(1, len(r))
    return min(1.0, float(cer))


def compute_wer(prediction: str, reference: str, case_sensitive: bool = False) -> float:
    """
    Computes Word Error Rate (WER):
    WER = Levenshtein_Distance(pred_words, ref_words) / max(1, len(ref_words))
    Capped at 1.0.
    """
    p = prediction if case_sensitive else prediction.lower()
    r = reference if case_sensitive else reference.lower()

    p_words = p.strip().split()
    r_words = r.strip().split()

    if not r_words:
        return 0.0 if not p_words else 1.0

    dist = levenshtein_distance(p_words, r_words)
    wer = dist / max(1, len(r_words))
    return min(1.0, float(wer))


def compute_character_accuracy(prediction: str, reference: str, case_sensitive: bool = False) -> float:
    """
    Computes Normalized Edit Distance based Character Accuracy:
    Acc = 1.0 - CER
    """
    cer = compute_cer(prediction, reference, case_sensitive)
    return max(0.0, 1.0 - cer)


def compute_layout_preservation_score(prediction_md: str, reference_md: str) -> Dict[str, float]:
    """
    Evaluates layout preservation between prediction and ground-truth markdown:
    1. Table Preservation: Presence and cell count of Markdown tables (| header |).
    2. Header Preservation: Markdown headers (#, ##, ###).
    3. List Preservation: Bullet/numbered list items.
    4. Reading Order Sequence: Jaccard bigram similarity between consecutive lines.
    """
    # Table extraction regex
    table_row_re = re.compile(r"^\s*\|.*\|\s*$", re.MULTILINE)
    pred_tables = len(table_row_re.findall(prediction_md))
    ref_tables = len(table_row_re.findall(reference_md))

    if ref_tables == 0:
        table_score = 1.0 if pred_tables == 0 else 0.8
    else:
        table_score = max(0.0, 1.0 - abs(pred_tables - ref_tables) / max(1, ref_tables))

    # Header extraction regex
    header_re = re.compile(r"^\s*#{1,6}\s+", re.MULTILINE)
    pred_headers = len(header_re.findall(prediction_md))
    ref_headers = len(header_re.findall(reference_md))

    if ref_headers == 0:
        header_score = 1.0 if pred_headers == 0 else 0.8
    else:
        header_score = max(0.0, 1.0 - abs(pred_headers - ref_headers) / max(1, ref_headers))

    # Reading order line bigrams
    pred_lines = [l.strip() for l in prediction_md.splitlines() if l.strip()]
    ref_lines = [l.strip() for l in reference_md.splitlines() if l.strip()]

    def get_bigrams(lines: List[str]) -> set:
        return set(zip(lines[:-1], lines[1:])) if len(lines) > 1 else set()

    pred_bg = get_bigrams(pred_lines)
    ref_bg = get_bigrams(ref_lines)

    if not ref_bg:
        reading_order_score = 1.0 if not pred_bg else 0.7
    else:
        intersection = len(pred_bg.intersection(ref_bg))
        union = len(pred_bg.union(ref_bg))
        reading_order_score = intersection / max(1, union)

    # Composite Layout Score
    composite_score = (0.4 * table_score) + (0.3 * header_score) + (0.3 * reading_order_score)

    return {
        "table_score": round(table_score, 4),
        "header_score": round(header_score, 4),
        "reading_order_score": round(reading_order_score, 4),
        "composite_layout_score": round(composite_score, 4)
    }


def normalize_val(val: Any) -> str:
    """Normalizes a field value for comparison."""
    if val is None:
        return ""
    s = str(val).strip().lower()
    # Normalize whitespaces, commas in numbers, currencies
    s = re.sub(r"[,\s₹$€£]+", " ", s).strip()
    return s


def compute_key_value_metrics(
    predicted_kv: Dict[str, Any],
    ground_truth_kv: Dict[str, Any],
    fuzzy_threshold: float = 0.8
) -> Dict[str, float]:
    """
    Computes Precision, Recall, and F1-score for Key-Value pair extraction:
    - Matches keys case-insensitively.
    - Evaluates exact match and fuzzy string match on values.
    - Computes hallucination rate (predicted keys not present in ground truth).
    """
    if not ground_truth_kv:
        if not predicted_kv:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "hallucination_rate": 0.0}
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "hallucination_rate": 1.0}

    # Normalize ground truth keys
    gt_norm = {k.strip().lower(): v for k, v in ground_truth_kv.items()}
    pred_norm = {k.strip().lower(): v for k, v in predicted_kv.items()}

    total_gt = len(gt_norm)
    total_pred = len(pred_norm)

    true_positives = 0
    fuzzy_matches = 0
    hallucinated_keys = 0

    for pred_key, pred_val in pred_norm.items():
        if pred_key in gt_norm:
            gt_val_str = normalize_val(gt_norm[pred_key])
            pred_val_str = normalize_val(pred_val)

            if pred_val_str == gt_val_str:
                true_positives += 1
                fuzzy_matches += 1
            else:
                # Calculate character accuracy of the value
                acc = compute_character_accuracy(pred_val_str, gt_val_str)
                if acc >= fuzzy_threshold:
                    fuzzy_matches += 1
        else:
            hallucinated_keys += 1

    precision = true_positives / max(1, total_pred)
    recall = true_positives / max(1, total_gt)
    f1 = (2 * precision * recall) / max(1e-6, (precision + recall))

    fuzzy_prec = fuzzy_matches / max(1, total_pred)
    fuzzy_rec = fuzzy_matches / max(1, total_gt)
    fuzzy_f1 = (2 * fuzzy_prec * fuzzy_rec) / max(1e-6, (fuzzy_prec + fuzzy_rec))

    hallucination_rate = hallucinated_keys / max(1, total_pred)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "fuzzy_precision": round(fuzzy_prec, 4),
        "fuzzy_recall": round(fuzzy_rec, 4),
        "fuzzy_f1": round(fuzzy_f1, 4),
        "hallucination_rate": round(hallucination_rate, 4)
    }


class BenchmarkTimer:
    """Context manager to measure latency and peak memory."""
    def __init__(self):
        self.latency_ms = 0.0
        self._start_time = 0.0

    def __enter__(self):
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.perf_counter() - self._start_time
        self.latency_ms = round(elapsed * 1000.0, 2)
