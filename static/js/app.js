/* Tenunnyo — app.js */
(function () {
  'use strict';

  const dropzone   = document.getElementById('dropzone');
  const fileInput  = document.getElementById('fileInput');
  const dzInner    = document.getElementById('dropzoneInner');
  const dzPreview  = document.getElementById('dzPreview');
  const previewImg = document.getElementById('previewImg');
  const btnRemove  = document.getElementById('btnRemove');
  const btnPredict = document.getElementById('btnPredict');
  const btnText    = document.getElementById('btnText');
  const btnLoading = document.getElementById('btnLoading');

  const resultEmpty   = document.getElementById('resultEmpty');
  const resultContent = document.getElementById('resultContent');

  let selectedFile = null;

  // ── Drag & Drop ──────────────────────────────────────────────
  ['dragenter','dragover'].forEach(e =>
    dropzone.addEventListener(e, ev => { ev.preventDefault(); dropzone.classList.add('drag-over'); })
  );
  ['dragleave','drop'].forEach(e =>
    dropzone.addEventListener(e, ev => { ev.preventDefault(); dropzone.classList.remove('drag-over'); })
  );
  dropzone.addEventListener('drop', ev => {
    const file = ev.dataTransfer?.files?.[0];
    if (file) setFile(file);
  });
  dropzone.addEventListener('click', e => {
    if (!e.target.closest('.btn-remove') && !e.target.closest('.btn-upload')) {
      fileInput.click();
    }
  });
  fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) setFile(fileInput.files[0]);
  });

  // ── Set file ─────────────────────────────────────────────────
  function setFile(file) {
    const allowed = ['image/png','image/jpeg','image/gif','image/bmp','image/webp'];
    if (!allowed.includes(file.type)) {
      showToast('Format file tidak didukung. Gunakan PNG, JPG, atau WEBP.');
      return;
    }
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = e => {
      previewImg.src = e.target.result;
      dzInner.style.display = 'none';
      dzPreview.style.display = 'block';
    };
    reader.readAsDataURL(file);
    btnPredict.disabled = false;
    clearResult();
  }

  // ── Remove ───────────────────────────────────────────────────
  btnRemove.addEventListener('click', e => {
    e.stopPropagation();
    selectedFile = null;
    fileInput.value = '';
    previewImg.src = '';
    dzInner.style.display = '';
    dzPreview.style.display = 'none';
    btnPredict.disabled = true;
    clearResult();
  });

  // ── Load sample ───────────────────────────────────────────────
  window.loadSample = function(url, name) {
    fetch(url)
      .then(r => r.blob())
      .then(blob => {
        const ext = url.split('.').pop();
        const file = new File([blob], `${name}.${ext}`, { type: `image/${ext}` });
        setFile(file);
      })
      .catch(() => showToast('Gagal memuat contoh gambar.'));
  };

  // ── Predict ───────────────────────────────────────────────────
  btnPredict.addEventListener('click', () => {
    if (!selectedFile) return;
    btnText.style.display = 'none';
    btnLoading.style.display = '';
    btnPredict.disabled = true;

    const fd = new FormData();
    fd.append('image', selectedFile);

    fetch('/predict', { method: 'POST', body: fd })
      .then(r => r.json())
      .then(data => {
        if (data.error) { showToast(data.error); return; }
        showResult(data);
      })
      .catch(() => showToast('Terjadi kesalahan jaringan.'))
      .finally(() => {
        btnText.style.display = '';
        btnLoading.style.display = 'none';
        btnPredict.disabled = false;
      });
  });

  // ── Show result ───────────────────────────────────────────────
  function showResult(data) {
    resultEmpty.style.display = 'none';
    resultContent.style.display = '';

    // Name & icon
    document.getElementById('resultName').textContent = data.prediction;
    document.getElementById('resultIcon').textContent = data.info?.icon || '🧵';

    // Info
    document.getElementById('infoOrigin').textContent  = data.info?.origin  || '—';
    document.getElementById('infoPattern').textContent = data.info?.pattern || '—';
    document.getElementById('infoDesc').textContent    = data.info?.desc    || '';

    // Confidence circle
    animateConfidence(data.confidence);

    // Probability bars
    renderBars(data.class_probs, data.prediction);
  }

  function animateConfidence(pct) {
    const arc = document.getElementById('confArc');
    const val = document.getElementById('confValue');
    const circ = 163.4;
    const offset = circ - (circ * pct / 100);

    // Animate number
    let start = 0, duration = 900;
    const t0 = performance.now();
    function tick(now) {
      const elapsed = now - t0;
      const progress = Math.min(elapsed / duration, 1);
      const ease = 1 - Math.pow(1 - progress, 3);
      val.textContent = Math.round(start + (pct - start) * ease) + '%';
      arc.style.strokeDashoffset = circ - (circ * ease * pct / 100);
      if (progress < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  }

  function renderBars(classProbs, topClass) {
    const container = document.getElementById('probaBars');
    container.innerHTML = '';
    classProbs.forEach((cp, i) => {
      const isTop = cp.name === topClass;
      const div = document.createElement('div');
      div.className = 'proba-item';
      div.innerHTML = `
        <div class="proba-meta">
          <span class="proba-name">${cp.name}</span>
          <span class="proba-pct">${cp.prob}%</span>
        </div>
        <div class="proba-bar-track">
          <div class="proba-bar-fill${isTop ? ' top-class' : ''}" data-w="${cp.prob}" style="width:0"></div>
        </div>`;
      container.appendChild(div);
    });
    // Animate bars
    requestAnimationFrame(() => {
      container.querySelectorAll('.proba-bar-fill').forEach(bar => {
        const w = bar.dataset.w;
        setTimeout(() => { bar.style.width = w + '%'; }, 100);
      });
    });
  }

  function clearResult() {
    resultContent.style.display = 'none';
    resultEmpty.style.display = '';
  }

  // ── Toast ─────────────────────────────────────────────────────
  function showToast(msg) {
    let tc = document.querySelector('.toast-container');
    if (!tc) {
      tc = document.createElement('div');
      tc.className = 'toast-container';
      document.body.appendChild(tc);
    }
    const t = document.createElement('div');
    t.className = 'toast-msg';
    t.textContent = msg;
    tc.appendChild(t);
    setTimeout(() => t.remove(), 4000);
  }

})();
