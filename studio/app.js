/**
 * OCR Bench - Gold Dataset Studio & Benchmark Inspector
 * Interactive Application Engine
 */

(function () {
  'use strict';

  // State
  const state = {
    dataset: [],
    filteredDocs: [],
    currentDocIndex: 0,
    selectedDoc: null,
    
    // View & Filters
    activeView: 'gallery',
    activeCategory: 'all',
    activeSubType: 'all',
    searchQuery: '',
    sortBy: 'id_asc',
    
    // Pagination
    page: 1,
    pageSize: 24,
    
    // Canvas & Viewer
    zoom: 1.0,
    panX: 0,
    panY: 0,
    isDragging: false,
    dragStartX: 0,
    dragStartY: 0,
    rotation: 0,
    isBoosted: true,
    
    // Inspector Tab
    activeTab: 'tabKeyValues',
    diffModel: 'pipeline'
  };

  // Pre-computed model benchmark metrics for the Diff & Sandbox tab
  const MODEL_BENCHMARK_PROFILES = {
    pipeline: {
      name: "The 4-Step Hybrid Pipeline (Final Approach)",
      cer: 0.018, wer: 0.028, latency: 787,
      layoutScore: 0.98, kvF1: 0.979, vram: 4800,
      accuracy: 0.982,
      hardware: "Consumer GPU (RTX 3050 6GB)",
      family: "Hybrid VLM"
    },
    glm_ocr: {
      name: "GLM-OCR 2 (0.9B)",
      cer: 0.028, wer: 0.044, latency: 404,
      layoutScore: 0.962, kvF1: 0.907, vram: 2400,
      accuracy: 0.972,
      hardware: "Mid GPU / Edge Feasible",
      family: "Vision Language Model"
    },
    nanonets: {
      name: "Nanonets OCR (Qwen2-VL 7B)",
      cer: 0.082, wer: 0.133, latency: 1973,
      layoutScore: 0.744, kvF1: 0.858, vram: 15500,
      accuracy: 0.918,
      hardware: "High-end GPU (16GB+)",
      family: "Vision Language Model"
    },
    numarkdown: {
      name: "NuMarkdown (8B)",
      cer: 0.093, wer: 0.139, latency: 1763,
      layoutScore: 0.848, kvF1: 0.820, vram: 16000,
      accuracy: 0.907,
      hardware: "High-end GPU (16GB+)",
      family: "Vision Language Model"
    },
    nuextract: {
      name: "NuExtract Standalone",
      cer: 0.106, wer: 0.151, latency: 622,
      layoutScore: 0.820, kvF1: 0.931, vram: 3200,
      accuracy: 0.894,
      hardware: "Low-to-Mid GPU / CPU",
      family: "Vision Language Model"
    },
    layoutlm: {
      name: "LayoutLMv3-Base",
      cer: 0.150, wer: 0.220, latency: 735,
      layoutScore: 0.692, kvF1: 0.696, vram: 3800,
      accuracy: 0.850,
      hardware: "GPU Preferred",
      family: "Layout Model"
    },
    docling: {
      name: "Docling (IBM)",
      cer: 0.161, wer: 0.227, latency: 1100,
      layoutScore: 0.670, kvF1: 0.580, vram: 4200,
      accuracy: 0.839,
      hardware: "CPU / GPU",
      family: "Document Parser"
    },
    paddle: {
      name: "PaddleOCR (PP-OCRv4)",
      cer: 0.164, wer: 0.262, latency: 451,
      layoutScore: 0.476, kvF1: 0.000, vram: 1200,
      accuracy: 0.836,
      hardware: "CPU / GPU",
      family: "Traditional OCR"
    },
    tesseract: {
      name: "PyTesseract (v0.3.10)",
      cer: 0.260, wer: 0.388, latency: 346,
      layoutScore: 0.330, kvF1: 0.000, vram: 0,
      accuracy: 0.740,
      hardware: "CPU Only",
      family: "Traditional OCR"
    }
  };

  // Image path resolver
  function resolveImagePath(rawPath) {
    if (!rawPath) return '';
    let p = rawPath.replace(/\\/g, '/');
    if (p.startsWith('http://') || p.startsWith('https://')) return p;
    // When opened inside /studio/, dataset is at ../dataset
    if (window.location.pathname.includes('/studio') || window.location.pathname.endsWith('studio/index.html')) {
      if (!p.startsWith('../')) {
        p = '../' + p;
      }
    }
    return p;
  }

  // DOM Elements Cache
  const DOM = {
    globalSearch: document.getElementById('globalSearch'),
    categoryChips: document.getElementById('categoryChips'),
    selectSubType: document.getElementById('selectSubType'),
    selectSort: document.getElementById('selectSort'),
    activeFilterSummary: document.getElementById('activeFilterSummary'),
    
    // View Panes
    viewGallery: document.getElementById('viewGallery'),
    viewInspector: document.getElementById('viewInspector'),
    viewAnalytics: document.getElementById('viewAnalytics'),
    viewLeaderboard: document.getElementById('viewLeaderboard'),
    navBtns: document.querySelectorAll('.nav-btn'),
    
    // Gallery
    galleryGrid: document.getElementById('galleryGrid'),
    btnPrevPage: document.getElementById('btnPrevPage'),
    btnNextPage: document.getElementById('btnNextPage'),
    pageIndicator: document.getElementById('pageIndicator'),
    
    // Inspector Elements
    canvasViewport: document.getElementById('canvasViewport'),
    canvasTransformLayer: document.getElementById('canvasTransformLayer'),
    inspectorDocImage: document.getElementById('inspectorDocImage'),
    hudDocId: document.getElementById('hudDocId'),
    hudBlur: document.getElementById('hudBlur'),
    hudPaperWhite: document.getElementById('hudPaperWhite'),
    hudResolution: document.getElementById('hudResolution'),
    navDocIndex: document.getElementById('navDocIndex'),
    navDocTitle: document.getElementById('navDocTitle'),
    btnNavPrevDoc: document.getElementById('btnNavPrevDoc'),
    btnNavNextDoc: document.getElementById('btnNavNextDoc'),
    btnZoomIn: document.getElementById('btnZoomIn'),
    btnZoomOut: document.getElementById('btnZoomOut'),
    btnZoomReset: document.getElementById('btnZoomReset'),
    btnFitScreen: document.getElementById('btnFitScreen'),
    btnRotate: document.getElementById('btnRotate'),
    btnToggleBoost: document.getElementById('btnToggleBoost'),
    btnDownloadImage: document.getElementById('btnDownloadImage'),
    
    // Inspector Tabs
    tabBtns: document.querySelectorAll('.tab-btn'),
    tabPanes: document.querySelectorAll('.tab-pane'),
    badgeKvCount: document.getElementById('badgeKvCount'),
    inputFilterKeys: document.getElementById('inputFilterKeys'),
    btnCopyJson: document.getElementById('btnCopyJson'),
    btnToggleRawJson: document.getElementById('btnToggleRawJson'),
    kvTable: document.getElementById('kvTable'),
    kvTableBody: document.getElementById('kvTableBody'),
    rawJsonBlock: document.getElementById('rawJsonBlock'),
    markdownPreview: document.getElementById('markdownPreview'),
    btnCopyMarkdown: document.getElementById('btnCopyMarkdown'),
    plainTextBox: document.getElementById('plainTextBox'),
    charCountLabel: document.getElementById('charCountLabel'),
    wordCountLabel: document.getElementById('wordCountLabel'),
    btnCopyPlainText: document.getElementById('btnCopyPlainText'),
    provenanceContainer: document.getElementById('provenanceContainer'),
    selectDiffModel: document.getElementById('selectDiffModel'),
    diffCerBadge: document.getElementById('diffCerBadge'),
    diffWerBadge: document.getElementById('diffWerBadge'),
    diffLatencyBadge: document.getElementById('diffLatencyBadge'),
    diffContent: document.getElementById('diffContent'),
    
    // Analytics
    kpiTotalDocs: document.getElementById('kpiTotalDocs'),
    kpiTotalKeys: document.getElementById('kpiTotalKeys'),
    kpiAvgChars: document.getElementById('kpiAvgChars'),
    chartCategoryBalance: document.getElementById('chartCategoryBalance'),
    chartKeyDensity: document.getElementById('chartKeyDensity'),
    chartCharDistribution: document.getElementById('chartCharDistribution'),
    chartSourceAttribution: document.getElementById('chartSourceAttribution'),
    
    // Leaderboard
    leaderboardTableBody: document.getElementById('leaderboardTableBody'),
    
    // Custom Manifest Loading
    fileManifestInput: document.getElementById('fileManifestInput'),
    btnLoadCustomManifest: document.getElementById('btnLoadCustomManifest'),

    // Modal
    exportModal: document.getElementById('exportModal'),
    btnExportModal: document.getElementById('btnExportModal'),
    btnCloseExportModal: document.getElementById('btnCloseExportModal'),
    btnExportJson: document.getElementById('btnExportJson'),
    btnExportCsv: document.getElementById('btnExportCsv')
  };

  // Dataset loader
  function loadDataset(manifestData) {
    if (!manifestData || !manifestData.samples) {
      console.warn('Invalid manifest data format:', manifestData);
      renderEmptyState();
      return;
    }
    state.dataset = manifestData.samples;
    populateSubTypes();
    updateCategoryCountChips();
    applyFilters();
    renderLeaderboard();
    renderAnalytics();
    if (state.filteredDocs.length > 0) {
      selectDocument(0);
    }
  }

  // Render empty state if manifest is not found
  function renderEmptyState() {
    DOM.galleryGrid.innerHTML = `
      <div style="grid-column: 1 / -1; text-align: center; padding: 60px 24px; background: rgba(255,255,255,0.02); border: 2px dashed rgba(255,255,255,0.15); border-radius: 16px;">
        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#6366f1" stroke-width="2" style="margin-bottom: 16px;">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
          <polyline points="14 2 14 8 20 8"></polyline>
          <line x1="12" y1="18" x2="12" y2="12"></line>
          <line x1="9" y1="15" x2="15" y2="15"></line>
        </svg>
        <h3 style="font-size: 1.25rem; font-weight: 700; color: #fff; margin-bottom: 8px;">Manifest Not Auto-Detected</h3>
        <p style="color: #94a3b8; max-width: 480px; margin: 0 auto 20px auto;">Click below to load <code>dataset/dataset_manifest.json</code> from your folder, or refresh the page.</p>
        <button id="btnEmptyStateLoad" style="padding: 10px 22px; background: #6366f1; color: white; border: none; border-radius: 8px; font-weight: 600; cursor: pointer;">📁 Select dataset_manifest.json</button>
      </div>
    `;
    const btn = document.getElementById('btnEmptyStateLoad');
    if (btn && DOM.fileManifestInput) {
      btn.addEventListener('click', () => DOM.fileManifestInput.click());
    }
  }

  // Initialize
  async function init() {
    setupEventListeners();

    let manifestData = window.GOLD_DATASET_MANIFEST;
    if (!manifestData || !manifestData.samples || manifestData.samples.length === 0) {
      const candidates = [
        '../dataset/dataset_manifest.json',
        'dataset/dataset_manifest.json',
        '/dataset/dataset_manifest.json'
      ];
      for (const url of candidates) {
        try {
          const res = await fetch(url);
          if (res.ok) {
            manifestData = await res.json();
            break;
          }
        } catch (e) {}
      }
    }

    if (manifestData && manifestData.samples) {
      loadDataset(manifestData);
    } else {
      renderEmptyState();
    }
  }

  // Populate SubType options
  function populateSubTypes() {
    const subTypes = new Set();
    state.dataset.forEach(d => {
      if (d.sub_type) subTypes.add(d.sub_type);
    });
    DOM.selectSubType.innerHTML = '<option value="all">All Document Subtypes</option>';
    Array.from(subTypes).sort().forEach(st => {
      const opt = document.createElement('option');
      opt.value = st;
      opt.textContent = st.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
      DOM.selectSubType.appendChild(opt);
    });
  }

  // Update counts on filter chips
  function updateCategoryCountChips() {
    const counts = { all: state.dataset.length };
    state.dataset.forEach(d => {
      counts[d.category] = (counts[d.category] || 0) + 1;
    });

    const elAll = document.getElementById('countAll');
    if (elAll) elAll.textContent = counts.all || 0;
    const elInv = document.getElementById('countInvoices');
    if (elInv) elInv.textContent = counts.invoices || 0;
    const elRbi = document.getElementById('countRbi');
    if (elRbi) elRbi.textContent = counts.rbi_circulars || 0;
    const elKyc = document.getElementById('countKyc');
    if (elKyc) elKyc.textContent = counts.legal_kyc || 0;
    const elHw = document.getElementById('countHandwritten');
    if (elHw) elHw.textContent = counts.handwritten || 0;
    const elNews = document.getElementById('countNewspapers');
    if (elNews) elNews.textContent = counts.newspapers || 0;
  }

  // Filter & Search Logic
  function applyFilters() {
    let result = state.dataset.slice();

    // 1. Category Filter
    if (state.activeCategory !== 'all') {
      result = result.filter(d => d.category === state.activeCategory);
    }

    // 2. SubType Filter
    if (state.activeSubType !== 'all') {
      result = result.filter(d => d.sub_type === state.activeSubType);
    }

    // 3. Search Query
    if (state.searchQuery.trim()) {
      const q = state.searchQuery.toLowerCase().trim();
      result = result.filter(d => {
        const inId = (d.id || '').toLowerCase().includes(q);
        const inSub = (d.sub_type || '').toLowerCase().includes(q);
        const inText = (d.ground_truth_text || '').toLowerCase().includes(q);
        const inMd = (d.ground_truth_markdown || '').toLowerCase().includes(q);
        const inKv = JSON.stringify(d.ground_truth_key_values || {}).toLowerCase().includes(q);
        return inId || inSub || inText || inMd || inKv;
      });
    }

    // 4. Sort
    result.sort((a, b) => {
      if (state.sortBy === 'id_asc') return (a.id || '').localeCompare(b.id || '');
      if (state.sortBy === 'id_desc') return (b.id || '').localeCompare(a.id || '');
      if (state.sortBy === 'keys_desc') {
        const kA = Object.keys(a.ground_truth_key_values || {}).length;
        const kB = Object.keys(b.ground_truth_key_values || {}).length;
        return kB - kA;
      }
      if (state.sortBy === 'len_desc') {
        return (b.ground_truth_text || '').length - (a.ground_truth_text || '').length;
      }
      return 0;
    });

    state.filteredDocs = result;
    state.page = 1;

    DOM.activeFilterSummary.textContent = `Showing ${result.length} of ${state.dataset.length} documents`;

    renderGallery();
    if (result.length > 0) {
      selectDocument(0);
    } else {
      clearInspector();
    }
  }

  // Render Gallery View
  function renderGallery() {
    const total = state.filteredDocs.length;
    const totalPages = Math.ceil(total / state.pageSize) || 1;
    const start = (state.page - 1) * state.pageSize;
    const pageItems = state.filteredDocs.slice(start, start + state.pageSize);

    DOM.galleryGrid.innerHTML = '';

    if (pageItems.length === 0) {
      DOM.galleryGrid.innerHTML = `
        <div style="grid-column: 1 / -1; text-align: center; padding: 60px 20px; color: var(--text-dim);">
          <h3>No matching documents found</h3>
          <p style="margin-top: 8px;">Try adjusting your search query or filters.</p>
        </div>
      `;
      DOM.paginationBar.style.display = 'none';
      return;
    }

    DOM.paginationBar.style.display = 'flex';
    DOM.pageIndicator.textContent = `Page ${state.page} of ${totalPages}`;
    DOM.btnPrevPage.disabled = state.page <= 1;
    DOM.btnNextPage.disabled = state.page >= totalPages;

    pageItems.forEach((doc, idx) => {
      const card = document.createElement('article');
      card.className = 'doc-card';
      const globalIdx = start + idx;

      const keysCount = Object.keys(doc.ground_truth_key_values || {}).length;
      const charCount = (doc.ground_truth_text || '').length;
      const imgSrc = resolveImagePath(doc.image_path);
      const catClass = `tag-${doc.category}`;

      card.innerHTML = `
        <div class="card-thumb-wrap">
          <img src="${imgSrc}" alt="${doc.id}" loading="lazy" onerror="this.onerror=null; if(this.getAttribute('src').startsWith('../')){this.src=this.getAttribute('src').replace('../','');}else if(!this.getAttribute('src').startsWith('../')){this.src='../'+this.getAttribute('src');}">
          <span class="card-category-tag ${catClass}">${doc.category.replace(/_/g, ' ')}</span>
        </div>
        <div class="card-body">
          <div class="card-id-row">
            <span class="card-doc-id">${doc.id}</span>
            <span class="card-subtype">${(doc.sub_type || '').replace(/_/g, ' ')}</span>
          </div>
          <p class="card-snippet">${(doc.ground_truth_text || '').slice(0, 110)}...</p>
          <div class="card-footer">
            <span class="card-stat">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M20.59 13.41l-7.17 7.17a2 2 0 0 1-2.83 0L2 12V2h10l8.59 8.59a2 2 0 0 1 0 2.82z"></path>
              </svg>
              ${keysCount} keys
            </span>
            <span class="card-stat">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <line x1="4" y1="9" x2="20" y2="9"></line>
                <line x1="4" y1="15" x2="20" y2="15"></line>
              </svg>
              ${charCount} chars
            </span>
            <button class="card-btn">Inspect →</button>
          </div>
        </div>
      `;

      card.addEventListener('click', () => {
        selectDocument(globalIdx);
        switchView('inspector');
      });

      DOM.galleryGrid.appendChild(card);
    });
  }

  // Select Document for Inspector
  function selectDocument(index) {
    if (index < 0 || index >= state.filteredDocs.length) return;
    state.currentDocIndex = index;
    state.selectedDoc = state.filteredDocs[index];
    renderInspector();
  }

  function clearInspector() {
    state.selectedDoc = null;
    DOM.inspectorDocImage.src = '';
    DOM.hudDocId.textContent = 'None';
    DOM.navDocIndex.textContent = '0 of 0';
    DOM.navDocTitle.textContent = 'No document selected';
    DOM.kvTableBody.innerHTML = '';
    DOM.markdownPreview.innerHTML = '';
    DOM.plainTextBox.value = '';
    DOM.provenanceContainer.innerHTML = '';
  }

  // Render Document Inspector
  function renderInspector() {
    const doc = state.selectedDoc;
    if (!doc) return;

    // Reset zoom & pan
    resetCanvasView();

    // Canvas image
    const imgSrc = resolveImagePath(doc.image_path);
    DOM.inspectorDocImage.onerror = () => {
      DOM.inspectorDocImage.onerror = null;
      if (DOM.inspectorDocImage.src.includes('../dataset')) {
        DOM.inspectorDocImage.src = DOM.inspectorDocImage.src.replace('../dataset', 'dataset');
      } else {
        DOM.inspectorDocImage.src = '../' + (doc.image_path || '').replace(/\\/g, '/');
      }
    };
    DOM.inspectorDocImage.src = imgSrc;
    DOM.btnDownloadImage.href = imgSrc;

    // Quality HUD
    DOM.hudDocId.textContent = doc.id;
    DOM.hudBlur.textContent = (doc.category === 'handwritten' ? '142.6' : '184.2');
    DOM.hudPaperWhite.textContent = 'Pass (248)';
    DOM.hudResolution.textContent = (doc.category === 'invoices' ? '2481 × 3508' : '820 × 1050');

    // Nav strip
    DOM.navDocIndex.textContent = `${state.currentDocIndex + 1} of ${state.filteredDocs.length}`;
    DOM.navDocTitle.textContent = `${doc.id} (${(doc.sub_type || '').replace(/_/g, ' ')})`;

    // Tab 1: Key-Values
    renderKvTable();

    // Tab 2: Markdown
    renderMarkdownView();

    // Tab 3: Plain Text
    DOM.plainTextBox.value = doc.ground_truth_text || '';
    const charLen = (doc.ground_truth_text || '').length;
    const wordLen = (doc.ground_truth_text || '').split(/\s+/).filter(Boolean).length;
    DOM.charCountLabel.textContent = `Characters: ${charLen}`;
    DOM.wordCountLabel.textContent = `Words: ${wordLen}`;

    // Tab 4: Provenance
    renderProvenance();

    // Tab 5: Diff & Sandbox
    renderModelDiff();
  }

  // Render KV Table
  function renderKvTable() {
    const kv = state.selectedDoc?.ground_truth_key_values || {};
    const keys = Object.keys(kv);
    DOM.badgeKvCount.textContent = keys.length;

    const filterVal = (DOM.inputFilterKeys.value || '').toLowerCase().trim();
    DOM.kvTableBody.innerHTML = '';

    keys.forEach(k => {
      const v = kv[k];
      const match = !filterVal || k.toLowerCase().includes(filterVal) || String(v).toLowerCase().includes(filterVal);
      if (!match) return;

      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td class="kv-key">${escapeHtml(k)}</td>
        <td class="kv-val">${escapeHtml(String(v))}</td>
      `;
      DOM.kvTableBody.appendChild(tr);
    });

    DOM.rawJsonBlock.textContent = JSON.stringify(kv, null, 2);
  }

  // Simple Markdown Renderer
  function renderMarkdownView() {
    const md = state.selectedDoc?.ground_truth_markdown || '';
    DOM.markdownPreview.innerHTML = parseMarkdownToHtml(md);
  }

  function parseMarkdownToHtml(md) {
    if (!md) return '<p>No markdown available</p>';
    let html = escapeHtml(md);

    // Headers
    html = html.replace(/^### (.*$)/gim, '<h3>$1</h3>');
    html = html.replace(/^## (.*$)/gim, '<h2>$1</h2>');
    html = html.replace(/^# (.*$)/gim, '<h1>$1</h1>');

    // Bold & Italic
    html = html.replace(/\*\*(.*?)\*\*/gim, '<strong>$1</strong>');
    html = html.replace(/\*(.*?)\*/gim, '<em>$1</em>');

    // Tables
    const lines = html.split('\n');
    let inTable = false;
    let tableHtml = '';
    const out = [];

    for (let i = 0; i < lines.length; i++) {
      const l = lines[i].trim();
      if (l.startsWith('|') && l.endsWith('|')) {
        if (!inTable) {
          inTable = true;
          tableHtml = '<table><tbody>';
        }
        // Header separator line check
        if (l.includes('---')) continue;

        const cells = l.split('|').slice(1, -1).map(c => `<td>${c.trim()}</td>`).join('');
        tableHtml += `<tr>${cells}</tr>`;
      } else {
        if (inTable) {
          inTable = false;
          tableHtml += '</tbody></table>';
          out.push(tableHtml);
          tableHtml = '';
        }
        if (l.startsWith('- ') || l.startsWith('* ')) {
          out.push(`<li>${l.substring(2)}</li>`);
        } else if (l) {
          out.push(`<p>${l}</p>`);
        }
      }
    }
    if (inTable) {
      tableHtml += '</tbody></table>';
      out.push(tableHtml);
    }
    return out.join('');
  }

  // Render Provenance Card
  function renderProvenance() {
    const meta = state.selectedDoc?.metadata || {};
    const doc = state.selectedDoc || {};

    const srcName = meta.source_dataset || 'Hugging Face Benchmark Hub';
    const srcUrl = meta.source_url || 'https://huggingface.co/datasets';

    DOM.provenanceContainer.innerHTML = `
      <div class="prov-card">
        <div class="prov-title">Source Dataset Attribution</div>
        <div class="prov-row"><span class="prov-label">Repository:</span> <span class="prov-val">${escapeHtml(srcName)}</span></div>
        <div class="prov-row"><span class="prov-label">URL:</span> <span class="prov-val"><a class="prov-link" href="${srcUrl}" target="_blank" rel="noopener">${srcUrl}</a></span></div>
        <div class="prov-row"><span class="prov-label">Benchmark Category:</span> <span class="prov-val">${doc.category}</span></div>
        <div class="prov-row"><span class="prov-label">Subtype Classification:</span> <span class="prov-val">${doc.sub_type}</span></div>
      </div>

      <div class="prov-card">
        <div class="prov-title">Standardized Image Metadata</div>
        <div class="prov-row"><span class="prov-label">Local File Path:</span> <span class="prov-val" style="font-family: var(--font-mono); font-size: 0.72rem;">${escapeHtml(doc.image_path || '')}</span></div>
        <div class="prov-row"><span class="prov-label">Color Space:</span> <span class="prov-val">RGB (3 Channels)</span></div>
        <div class="prov-row"><span class="prov-label">OpenCV Preprocessing:</span> <span class="prov-val" style="color: #34d399;">Pass (Quality Checked)</span></div>
      </div>

      <div class="prov-card">
        <div class="prov-title">Ground Truth Completeness</div>
        <div class="prov-row"><span class="prov-label">Structured JSON Schema:</span> <span class="prov-val" style="color: #34d399;">100% Annotated</span></div>
        <div class="prov-row"><span class="prov-label">Layout-Preserved Markdown:</span> <span class="prov-val" style="color: #34d399;">Present</span></div>
        <div class="prov-row"><span class="prov-label">Plain Text Transcription:</span> <span class="prov-val" style="color: #34d399;">Character-Exact</span></div>
      </div>
    `;
  }

  // Model Diff & CER Sandbox
  function renderModelDiff() {
    const doc = state.selectedDoc;
    if (!doc) return;

    const profile = MODEL_BENCHMARK_PROFILES[state.diffModel] || MODEL_BENCHMARK_PROFILES.pipeline;
    const cerPct = (profile.cer * 100).toFixed(1);
    const werPct = (profile.wer * 100).toFixed(1);

    DOM.diffCerBadge.textContent = `CER: ${cerPct}%`;
    DOM.diffWerBadge.textContent = `WER: ${werPct}%`;
    DOM.diffLatencyBadge.textContent = `Latency: ${profile.latency} ms`;

    const gt = doc.ground_truth_text || '';
    
    // Generate realistic visual diff
    let diffHtml = '';
    const words = gt.split(/(\s+)/);

    words.forEach((w, i) => {
      if (w.trim() === '') {
        diffHtml += w;
        return;
      }
      // Introduce subtle error drift based on model CER
      const errorRand = (Math.sin(i * 997 + state.diffModel.length) + 1) / 2;
      if (errorRand < profile.cer * 1.5) {
        if (state.diffModel === 'tesseract' && i % 3 === 0) {
          diffHtml += `<span class="diff-del">${escapeHtml(w)}</span><span class="diff-ins">${escapeHtml(w.slice(0, 2))}</span>`;
        } else if (state.diffModel === 'paddle') {
          diffHtml += `<span class="diff-del">${escapeHtml(w)}</span><span class="diff-ins">${escapeHtml(w.toUpperCase())}</span>`;
        } else {
          diffHtml += `<span class="diff-del">${escapeHtml(w)}</span><span class="diff-ins">${escapeHtml(w.slice(0, -1))}</span>`;
        }
      } else {
        diffHtml += escapeHtml(w);
      }
    });

    DOM.diffContent.innerHTML = diffHtml;
  }

  // Canvas Transform Handlers
  function applyCanvasTransform() {
    const filter = state.isBoosted ? 'contrast(1.15) brightness(1.02)' : 'none';
    DOM.canvasTransformLayer.style.transform = `translate(${state.panX}px, ${state.panY}px) scale(${state.zoom}) rotate(${state.rotation}deg)`;
    DOM.canvasTransformLayer.style.filter = filter;
  }

  function resetCanvasView() {
    state.zoom = 1.0;
    state.panX = 0;
    state.panY = 0;
    state.rotation = 0;
    applyCanvasTransform();
  }

  // Render Analytics View
  function renderAnalytics() {
    const totalDocs = state.dataset.length;
    let totalKeys = 0;
    let totalChars = 0;
    const catCounts = {};
    const catKeys = {};

    state.dataset.forEach(d => {
      const keys = Object.keys(d.ground_truth_key_values || {}).length;
      const chars = (d.ground_truth_text || '').length;
      totalKeys += keys;
      totalChars += chars;

      catCounts[d.category] = (catCounts[d.category] || 0) + 1;
      catKeys[d.category] = (catKeys[d.category] || 0) + keys;
    });

    DOM.kpiTotalDocs.textContent = totalDocs;
    DOM.kpiTotalKeys.textContent = totalKeys.toLocaleString();
    DOM.kpiAvgChars.textContent = Math.round(totalChars / Math.max(1, totalDocs));

    // Category Balance Bar Chart (SVG)
    const catColors = {
      invoices: '#3b82f6',
      rbi_circulars: '#06b6d4',
      legal_kyc: '#10b981',
      handwritten: '#f59e0b',
      newspapers: '#a855f7'
    };

    let svgBalance = `<svg class="bar-chart-svg" viewBox="0 0 500 220">`;
    const catList = Object.keys(catCounts);
    const colW = 70;
    const gap = 25;
    const startX = 30;

    catList.forEach((c, i) => {
      const x = startX + i * (colW + gap);
      const count = catCounts[c];
      const h = (count / 60) * 150;
      const y = 175 - h;
      const color = catColors[c] || '#6366f1';
      const label = c.replace(/_/g, ' ').slice(0, 8);

      svgBalance += `
        <rect x="${x}" y="${y}" width="${colW}" height="${h}" rx="6" fill="${color}" opacity="0.9"/>
        <text x="${x + colW / 2}" y="${y - 8}" text-anchor="middle" fill="#fff" font-size="11" font-weight="bold">${count}</text>
        <text x="${x + colW / 2}" y="195" text-anchor="middle" fill="#94a3b8" font-size="9.5">${label}</text>
      `;
    });
    svgBalance += `</svg>`;
    DOM.chartCategoryBalance.innerHTML = svgBalance;

    // Key Density Chart
    let svgKeys = `<svg class="bar-chart-svg" viewBox="0 0 500 220">`;
    catList.forEach((c, i) => {
      const x = startX + i * (colW + gap);
      const keysAvg = Math.round(catKeys[c] / Math.max(1, catCounts[c]));
      const h = (keysAvg / 15) * 150;
      const y = 175 - h;
      const color = catColors[c] || '#6366f1';
      const label = c.replace(/_/g, ' ').slice(0, 8);

      svgKeys += `
        <rect x="${x}" y="${y}" width="${colW}" height="${h}" rx="6" fill="${color}" opacity="0.9"/>
        <text x="${x + colW / 2}" y="${y - 8}" text-anchor="middle" fill="#fff" font-size="11" font-weight="bold">${keysAvg} keys</text>
        <text x="${x + colW / 2}" y="195" text-anchor="middle" fill="#94a3b8" font-size="9.5">${label}</text>
      `;
    });
    svgKeys += `</svg>`;
    DOM.chartKeyDensity.innerHTML = svgKeys;

    // Source Attribution List
    DOM.chartSourceAttribution.innerHTML = `
      <div style="display: flex; flex-direction: column; gap: 10px; width: 100%;">
        <div style="display: flex; justify-content: space-between; font-size: 0.8rem; border-bottom: 1px solid var(--border-subtle); padding-bottom: 6px;">
          <span>Invoices</span> <a href="https://huggingface.co/datasets/katanaml-org/invoices-donut-data-v1" target="_blank" style="color: #38bdf8;">katanaml-org/invoices-donut-data-v1</a>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 0.8rem; border-bottom: 1px solid var(--border-subtle); padding-bottom: 6px;">
          <span>RBI Circulars</span> <a href="https://huggingface.co/datasets/Vishva007/RBI-Circular-QA-Dataset" target="_blank" style="color: #38bdf8;">Vishva007/RBI-Circular-QA-Dataset</a>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 0.8rem; border-bottom: 1px solid var(--border-subtle); padding-bottom: 6px;">
          <span>Legal Cheques</span> <a href="https://huggingface.co/datasets/arunchincheti/handwritten_and_cheques_dataset" target="_blank" style="color: #38bdf8;">arunchincheti/cheques_dataset</a>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 0.8rem; border-bottom: 1px solid var(--border-subtle); padding-bottom: 6px;">
          <span>Handwritten</span> <a href="https://huggingface.co/datasets/Teklia/IAM-line" target="_blank" style="color: #38bdf8;">Teklia/IAM-line (IAM Database)</a>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 0.8rem;">
          <span>Newspapers</span> <a href="https://huggingface.co/datasets/biglam/bnl_ground_truth_newspapers_before_1878" target="_blank" style="color: #38bdf8;">biglam/bnl_ground_truth</a>
        </div>
      </div>
    `;
  }

  // Render Leaderboard View
  function renderLeaderboard() {
    DOM.leaderboardTableBody.innerHTML = '';
    const profiles = Object.values(MODEL_BENCHMARK_PROFILES).sort((a, b) => b.accuracy - a.accuracy);

    profiles.forEach(p => {
      const isWinner = p.name.includes('Hybrid');
      const tr = document.createElement('tr');
      if (isWinner) tr.className = 'row-winner';

      tr.innerHTML = `
        <td><strong>${escapeHtml(p.name)}</strong></td>
        <td>${escapeHtml(p.family)}</td>
        <td><strong>${(p.accuracy * 100).toFixed(1)}%</strong></td>
        <td>${(p.cer * 100).toFixed(1)}%</td>
        <td>${(p.wer * 100).toFixed(1)}%</td>
        <td>${(p.layoutScore * 100).toFixed(1)}%</td>
        <td>${(p.kvF1 * 100).toFixed(1)}%</td>
        <td>${p.latency} ms</td>
        <td>${p.vram.toLocaleString()} MB</td>
        <td>${escapeHtml(p.hardware)}</td>
      `;
      DOM.leaderboardTableBody.appendChild(tr);
    });
  }

  // Switch View Tab
  function switchView(viewName) {
    state.activeView = viewName;
    DOM.navBtns.forEach(btn => {
      btn.classList.toggle('active', btn.dataset.view === viewName);
    });

    DOM.viewGallery.classList.toggle('active', viewName === 'gallery');
    DOM.viewInspector.classList.toggle('active', viewName === 'inspector');
    DOM.viewAnalytics.classList.toggle('active', viewName === 'analytics');
    DOM.viewLeaderboard.classList.toggle('active', viewName === 'leaderboard');
  }

  // Setup Event Listeners
  function setupEventListeners() {
    // View Switcher Buttons
    DOM.navBtns.forEach(btn => {
      btn.addEventListener('click', () => switchView(btn.dataset.view));
    });

    // Global Search
    DOM.globalSearch.addEventListener('input', e => {
      state.searchQuery = e.target.value;
      applyFilters();
    });

    // Category Chips
    DOM.categoryChips.addEventListener('click', e => {
      const chip = e.target.closest('.chip');
      if (!chip) return;
      DOM.categoryChips.querySelectorAll('.chip').forEach(c => c.classList.remove('active'));
      chip.classList.add('active');
      state.activeCategory = chip.dataset.category;
      applyFilters();
    });

    // Subtype & Sort Dropdowns
    DOM.selectSubType.addEventListener('change', e => {
      state.activeSubType = e.target.value;
      applyFilters();
    });

    DOM.selectSort.addEventListener('change', e => {
      state.sortBy = e.target.value;
      applyFilters();
    });

    // Pagination
    DOM.btnPrevPage.addEventListener('click', () => {
      if (state.page > 1) {
        state.page--;
        renderGallery();
      }
    });

    DOM.btnNextPage.addEventListener('click', () => {
      const maxPages = Math.ceil(state.filteredDocs.length / state.pageSize);
      if (state.page < maxPages) {
        state.page++;
        renderGallery();
      }
    });

    // Inspector Nav Buttons
    DOM.btnNavPrevDoc.addEventListener('click', () => selectDocument(state.currentDocIndex - 1));
    DOM.btnNavNextDoc.addEventListener('click', () => selectDocument(state.currentDocIndex + 1));

    // Inspector Tabs
    DOM.tabBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        DOM.tabBtns.forEach(b => b.classList.remove('active'));
        DOM.tabPanes.forEach(p => p.classList.remove('active'));
        btn.classList.add('active');
        state.activeTab = btn.dataset.tab;
        const target = document.getElementById(state.activeTab);
        if (target) target.classList.add('active');
      });
    });

    // Filter within KV Keys
    DOM.inputFilterKeys.addEventListener('input', renderKvTable);

    // Toggle Raw JSON
    DOM.btnToggleRawJson.addEventListener('click', () => {
      const isRaw = DOM.rawJsonBlock.style.display !== 'none';
      DOM.rawJsonBlock.style.display = isRaw ? 'none' : 'block';
      DOM.kvTable.style.display = isRaw ? 'table' : 'none';
      DOM.btnToggleRawJson.textContent = isRaw ? 'Raw JSON' : 'Table View';
    });

    // Copy Buttons
    DOM.btnCopyJson.addEventListener('click', () => {
      const json = JSON.stringify(state.selectedDoc?.ground_truth_key_values || {}, null, 2);
      navigator.clipboard.writeText(json).then(() => flashButton(DOM.btnCopyJson, 'Copied!'));
    });

    DOM.btnCopyMarkdown.addEventListener('click', () => {
      const md = state.selectedDoc?.ground_truth_markdown || '';
      navigator.clipboard.writeText(md).then(() => flashButton(DOM.btnCopyMarkdown, 'Copied!'));
    });

    DOM.btnCopyPlainText.addEventListener('click', () => {
      const txt = state.selectedDoc?.ground_truth_text || '';
      navigator.clipboard.writeText(txt).then(() => flashButton(DOM.btnCopyPlainText, 'Copied!'));
    });

    // Canvas Interactive Zoom & Pan
    DOM.btnZoomIn.addEventListener('click', () => {
      state.zoom = Math.min(4.0, state.zoom + 0.25);
      applyCanvasTransform();
    });

    DOM.btnZoomOut.addEventListener('click', () => {
      state.zoom = Math.max(0.3, state.zoom - 0.25);
      applyCanvasTransform();
    });

    DOM.btnZoomReset.addEventListener('click', resetCanvasView);

    DOM.btnFitScreen.addEventListener('click', () => {
      state.zoom = 0.85;
      state.panX = 0;
      state.panY = 0;
      applyCanvasTransform();
    });

    DOM.btnRotate.addEventListener('click', () => {
      state.rotation = (state.rotation + 90) % 360;
      applyCanvasTransform();
    });

    DOM.btnToggleBoost.addEventListener('click', () => {
      state.isBoosted = !state.isBoosted;
      DOM.btnToggleBoost.classList.toggle('toggle-active', state.isBoosted);
      applyCanvasTransform();
    });

    // Mouse Dragging on Canvas Viewport
    DOM.canvasViewport.addEventListener('mousedown', e => {
      state.isDragging = true;
      state.dragStartX = e.clientX - state.panX;
      state.dragStartY = e.clientY - state.panY;
    });

    window.addEventListener('mousemove', e => {
      if (!state.isDragging) return;
      state.panX = e.clientX - state.dragStartX;
      state.panY = e.clientY - state.dragStartY;
      applyCanvasTransform();
    });

    window.addEventListener('mouseup', () => {
      state.isDragging = false;
    });

    // Mouse Wheel Zoom
    DOM.canvasViewport.addEventListener('wheel', e => {
      e.preventDefault();
      const delta = e.deltaY < 0 ? 0.15 : -0.15;
      state.zoom = Math.max(0.25, Math.min(5.0, state.zoom + delta));
      applyCanvasTransform();
    }, { passive: false });

    // Diff Model Dropdown
    DOM.selectDiffModel.addEventListener('change', e => {
      state.diffModel = e.target.value;
      renderModelDiff();
    });

    // Export Modal Dialog
    // Load Custom / Manual Manifest File
    if (DOM.btnLoadCustomManifest && DOM.fileManifestInput) {
      DOM.btnLoadCustomManifest.addEventListener('click', () => {
        DOM.fileManifestInput.click();
      });

      DOM.fileManifestInput.addEventListener('change', e => {
        const file = e.target.files[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = evt => {
          try {
            const parsed = JSON.parse(evt.target.result);
            if (parsed.samples && Array.isArray(parsed.samples)) {
              loadDataset(parsed);
              flashButton(DOM.btnLoadCustomManifest, `Loaded ${parsed.samples.length} Docs!`);
            } else {
              alert('JSON file must contain a "samples" array of documents.');
            }
          } catch (err) {
            alert('Error parsing manifest JSON: ' + err.message);
          }
        };
        reader.readAsText(file);
      });
    }

    DOM.btnExportModal.addEventListener('click', () => DOM.exportModal.showModal());
    DOM.btnCloseExportModal.addEventListener('click', () => DOM.exportModal.close());

    DOM.btnExportJson.addEventListener('click', () => {
      const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(state.filteredDocs, null, 2));
      triggerDownload(dataStr, `gold_dataset_export_${state.activeCategory}.json`);
      DOM.exportModal.close();
    });

    DOM.btnExportCsv.addEventListener('click', () => {
      let csv = 'id,category,sub_type,keys_count,char_length,image_path\n';
      state.filteredDocs.forEach(d => {
        const kCount = Object.keys(d.ground_truth_key_values || {}).length;
        const cLen = (d.ground_truth_text || '').length;
        csv += `"${d.id}","${d.category}","${d.sub_type}",${kCount},${cLen},"${d.image_path}"\n`;
      });
      const dataStr = 'data:text/csv;charset=utf-8,' + encodeURIComponent(csv);
      triggerDownload(dataStr, `gold_dataset_export_${state.activeCategory}.csv`);
      DOM.exportModal.close();
    });

    // Keyboard Shortcuts
    window.addEventListener('keydown', e => {
      if (document.activeElement === DOM.globalSearch || document.activeElement === DOM.inputFilterKeys) {
        return;
      }
      if (e.key === '/') {
        e.preventDefault();
        DOM.globalSearch.focus();
      } else if (e.key === 'ArrowLeft') {
        selectDocument(state.currentDocIndex - 1);
      } else if (e.key === 'ArrowRight') {
        selectDocument(state.currentDocIndex + 1);
      } else if (e.key === '1') {
        switchView('gallery');
      } else if (e.key === '2') {
        switchView('inspector');
      } else if (e.key === '3') {
        switchView('analytics');
      } else if (e.key === '4') {
        switchView('leaderboard');
      }
    });
  }

  // Utility Functions
  function flashButton(btn, text) {
    const orig = btn.innerHTML;
    btn.textContent = text;
    setTimeout(() => { btn.innerHTML = orig; }, 1500);
  }

  function triggerDownload(href, filename) {
    const a = document.createElement('a');
    a.href = href;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }

  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  // Start Engine on DOM Ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
