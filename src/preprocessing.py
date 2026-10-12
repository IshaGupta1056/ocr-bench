"""
src/preprocessing.py
Step 1: Document Preprocessing & OpenCV Image Quality Enhancement
- Extracts pagewise images from PDFs into temporary or target directories.
- Evaluates image quality: blur (Laplacian variance), contrast (RMS/Michelson),
  and background/foreground ink fidelity (paper-white & high-contrast black).
- Applies OpenCV enhancement pipeline for degraded, scanned, or handwritten documents.
- Encodes processed images to base64 for downstream VLM consumption.
"""

import os
import cv2
import base64
import tempfile
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Tuple, Dict, Any, Optional, Union

# Try importing PDF rendering engines
try:
    import pypdfium2 as pdfium
    HAS_PDFIUM = True
except ImportError:
    HAS_PDFIUM = False

try:
    from pdf2image import convert_from_path
    HAS_PDF2IMAGE = True
except ImportError:
    HAS_PDF2IMAGE = False


class ImageQualityMetrics:
    """Stores calculated image quality metrics."""
    def __init__(
        self,
        blur_score: float,
        is_blurry: bool,
        contrast_score: float,
        is_low_contrast: bool,
        bg_white_score: float,
        is_paper_white: bool,
        ink_black_score: float,
        is_high_contrast_black: bool,
        needs_enhancement: bool
    ):
        self.blur_score = blur_score
        self.is_blurry = is_blurry
        self.contrast_score = contrast_score
        self.is_low_contrast = is_low_contrast
        self.bg_white_score = bg_white_score
        self.is_paper_white = is_paper_white
        self.ink_black_score = ink_black_score
        self.is_high_contrast_black = is_high_contrast_black
        self.needs_enhancement = needs_enhancement

    def to_dict(self) -> Dict[str, Any]:
        return {
            "blur_score": round(self.blur_score, 2),
            "is_blurry": self.is_blurry,
            "contrast_score": round(self.contrast_score, 2),
            "is_low_contrast": self.is_low_contrast,
            "bg_white_score": round(self.bg_white_score, 2),
            "is_paper_white": self.is_paper_white,
            "ink_black_score": round(self.ink_black_score, 2),
            "is_high_contrast_black": self.is_high_contrast_black,
            "needs_enhancement": self.needs_enhancement
        }


class DocumentPreprocessor:
    """
    Production-grade document preprocessor implementing Step 1 of the OCR pipeline.
    """

    def __init__(
        self,
        blur_threshold: float = 120.0,
        contrast_threshold: float = 40.0,
        bg_white_threshold: float = 220.0,
        ink_black_threshold: float = 50.0,
        temp_dir: Optional[str] = None
    ):
        self.blur_threshold = blur_threshold
        self.contrast_threshold = contrast_threshold
        self.bg_white_threshold = bg_white_threshold
        self.ink_black_threshold = ink_black_threshold
        self.temp_dir = temp_dir or tempfile.gettempdir()

    def load_image(self, image_source: Union[str, Path, np.ndarray, Image.Image]) -> np.ndarray:
        """Loads an image into a BGR numpy array."""
        if isinstance(image_source, np.ndarray):
            if len(image_source.shape) == 2:
                return cv2.cvtColor(image_source, cv2.COLOR_GRAY2BGR)
            return image_source.copy()
        elif isinstance(image_source, Image.Image):
            rgb = np.array(image_source.convert("RGB"))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        elif isinstance(image_source, (str, Path)):
            path_str = str(image_source)
            if not os.path.exists(path_str):
                raise FileNotFoundError(f"Image file not found: {path_str}")
            # cv2.imread fails with unicode paths on Windows, use cv2.imdecode
            with open(path_str, "rb") as f:
                bytes_data = bytearray(f.read())
            img_array = np.asarray(bytes_data, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Failed to decode image from path: {path_str}")
            return img
        else:
            raise TypeError(f"Unsupported image input type: {type(image_source)}")

    def extract_pdf_pages(self, pdf_path: Union[str, Path], dpi: int = 300) -> List[np.ndarray]:
        """
        Extracts pagewise images from a PDF file.
        Uses pypdfium2 if available, falls back to pdf2image or fitz.
        """
        pdf_path = str(pdf_path)
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        images: List[np.ndarray] = []

        if HAS_PDFIUM:
            pdf = pdfium.PdfDocument(pdf_path)
            for i in range(len(pdf)):
                page = pdf[i]
                # Render at target scale
                scale = dpi / 72.0
                bitmap = page.render(scale=scale)
                pil_image = bitmap.to_pil()
                images.append(self.load_image(pil_image))
        elif HAS_PDF2IMAGE:
            pil_images = convert_from_path(pdf_path, dpi=dpi)
            for pil_img in pil_images:
                images.append(self.load_image(pil_img))
        else:
            raise RuntimeError(
                "Neither pypdfium2 nor pdf2image is installed. Please install 'pypdfium2'."
            )

        return images

    def check_quality(self, img: np.ndarray) -> ImageQualityMetrics:
        """
        Checks contrast, blur, paper-white background, and high-contrast black ink.
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # 1. Blur score: Variance of the Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur_score = float(laplacian.var())
        is_blurry = blur_score < self.blur_threshold

        # 2. Contrast score: Standard deviation of pixel intensities
        contrast_score = float(np.std(gray))
        is_low_contrast = contrast_score < self.contrast_threshold

        # 3. Background luminance (95th percentile should be near 255 for pure white paper)
        bg_white_score = float(np.percentile(gray, 95))
        is_paper_white = bg_white_score >= self.bg_white_threshold

        # 4. Foreground ink darkness (5th percentile should be dark, ideally <= 50)
        ink_black_score = float(np.percentile(gray, 5))
        is_high_contrast_black = ink_black_score <= self.ink_black_threshold

        # Needs enhancement if blurry, low contrast, yellowed/shadowed background, or faded ink
        needs_enhancement = (
            is_blurry or is_low_contrast or (not is_paper_white) or (not is_high_contrast_black)
        )

        return ImageQualityMetrics(
            blur_score=blur_score,
            is_blurry=is_blurry,
            contrast_score=contrast_score,
            is_low_contrast=is_low_contrast,
            bg_white_score=bg_white_score,
            is_paper_white=is_paper_white,
            ink_black_score=ink_black_score,
            is_high_contrast_black=is_high_contrast_black,
            needs_enhancement=needs_enhancement
        )

    def enhance_image(self, img: np.ndarray) -> np.ndarray:
        """
        Boosts quality using OpenCV:
        1. Morphological background division to flatten uneven shadows/yellow paper tint.
        2. CLAHE (Contrast Limited Adaptive Histogram Equalization) for localized text contrast.
        3. Unsharp masking to sharpen faint strokes and handwriting.
        4. Normalization to ensure pure paper-white [245-255] and deep ink black [0-25].
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # Step A: Estimate and divide out uneven illumination (morphological background estimate)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
        bg = cv2.morphologyEx(gray, cv2.MORPH_DILATE, kernel)
        bg = cv2.GaussianBlur(bg, (51, 51), 0)

        # Division normalization: (gray / bg) * 255
        norm = np.clip((gray.astype(np.float32) / (bg.astype(np.float32) + 1e-5)) * 255.0, 0, 255).astype(np.uint8)

        # Step B: Local contrast enhancement via CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        contrast_boost = clahe.apply(norm)

        # Step C: Unsharp Masking for handwriting & text edge crispness
        gaussian = cv2.GaussianBlur(contrast_boost, (0, 0), sigmaX=1.5)
        sharpened = cv2.addWeighted(contrast_boost, 1.4, gaussian, -0.4, 0)

        # Step D: Dynamic stretching to guarantee paper-white background
        # Pixels > 220 pushed towards 255; dark ink stretched
        p_low = np.percentile(sharpened, 2)
        p_high = np.percentile(sharpened, 98)
        stretched = np.clip((sharpened - p_low) / (p_high - p_low + 1e-5) * 255.0, 0, 255).astype(np.uint8)

        # Reconstruct 3-channel BGR for downstream VLM / OCR compatibility
        enhanced_bgr = cv2.cvtColor(stretched, cv2.COLOR_GRAY2BGR)
        return enhanced_bgr

    def process_document(
        self,
        document_path: Union[str, Path],
        force_enhancement: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Full Step 1 pipeline:
        - If PDF: extracts pages into images.
        - If image: processes directly.
        - Computes quality metrics.
        - Applies OpenCV enhancement if quality is subpar.
        - Returns list of pagewise dictionaries with images, metrics, and base64 strings.
        """
        doc_path = Path(document_path)
        if not doc_path.exists():
            raise FileNotFoundError(f"Document not found: {doc_path}")

        # Determine if PDF or image
        is_pdf = doc_path.suffix.lower() == ".pdf"
        if is_pdf:
            raw_pages = self.extract_pdf_pages(doc_path)
        else:
            raw_pages = [self.load_image(doc_path)]

        results: List[Dict[str, Any]] = []

        for page_idx, raw_img in enumerate(raw_pages):
            metrics = self.check_quality(raw_img)
            should_enhance = force_enhancement or metrics.needs_enhancement

            if should_enhance:
                processed_img = self.enhance_image(raw_img)
                enhanced_metrics = self.check_quality(processed_img)
            else:
                processed_img = raw_img
                enhanced_metrics = metrics

            b64_str = self.image_to_base64(processed_img)

            results.append({
                "page_number": page_idx + 1,
                "raw_image": raw_img,
                "processed_image": processed_img,
                "was_enhanced": should_enhance,
                "initial_quality": metrics.to_dict(),
                "final_quality": enhanced_metrics.to_dict(),
                "base64_image": b64_str
            })

        return results

    @staticmethod
    def image_to_base64(img: np.ndarray, ext: str = ".png") -> str:
        """Converts an OpenCV image array to a base64 encoded data string."""
        success, encoded = cv2.imencode(ext, img)
        if not success:
            raise ValueError("Failed to encode image to base64 format")
        b64_bytes = base64.b64encode(encoded)
        return f"data:image/png;base64,{b64_bytes.decode('utf-8')}"

    @staticmethod
    def base64_to_image(b64_str: str) -> np.ndarray:
        """Decodes base64 string back to OpenCV image array."""
        if "," in b64_str:
            b64_str = b64_str.split(",")[1]
        decoded = base64.b64decode(b64_str)
        img_array = np.frombuffer(decoded, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img
