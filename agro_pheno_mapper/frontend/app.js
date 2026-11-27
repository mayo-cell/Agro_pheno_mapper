const statusPill = document.getElementById("statusPill");
const plotNameInput = document.getElementById("plotName");
const gammaSlider = document.getElementById("gammaSlider");
const gammaValue = document.getElementById("gammaValue");
const sharpenToggle = document.getElementById("sharpenToggle");
const showEnhancedToggle = document.getElementById("showEnhancedToggle");

const imageInput = document.getElementById("imageInput");
const uploadBtn = document.getElementById("uploadBtn");
const sampleBtn = document.getElementById("sampleBtn");
const exportCsvBtn = document.getElementById("exportCsvBtn");
const refreshHistoryBtn = document.getElementById("refreshHistoryBtn");

const imgRaw = document.getElementById("imgRaw");
const imgHeatmap = document.getElementById("imgHeatmap");
const imgMask = document.getElementById("imgMask");
const phRaw = document.getElementById("phRaw");
const phHeatmap = document.getElementById("phHeatmap");
const phMask = document.getElementById("phMask");

let currentScan = null;

gammaSlider.addEventListener("input", () => {
  gammaValue.textContent = parseFloat(gammaSlider.value).toFixed(2);
});

showEnhancedToggle.addEventListener("change", () => {
  if (!currentScan) return;
  imgRaw.src = showEnhancedToggle.checked
    ? currentScan.enhanced_url
    : currentScan.raw_url;
});

uploadBtn.addEventListener("click", () => imageInput.click());

imageInput.addEventListener("change", async () => {
  const file = imageInput.files[0];
  if (!file) return;
  const form = new FormData();
  form.append("file", file);
  form.append("plot_name", plotNameInput.value.trim() || "Plot A-1 (Demo)");
  form.append("gamma", gammaSlider.value);
  form.append("sharpen", sharpenToggle.checked ? "true" : "false");

  await runAnalysis("/api/analyze", form);
  imageInput.value = "";
});

sampleBtn.addEventListener("click", async () => {
  const form = new FormData();
  form.append("plot_name", plotNameInput.value.trim() || "Plot A-1 (Demo)");
  form.append("gamma", gammaSlider.value);
  form.append("sharpen", sharpenToggle.checked ? "true" : "false");

  await runAnalysis("/api/analyze-sample", form);
});

exportCsvBtn.addEventListener("click", () => {
  if (!currentScan) return;
  window.location.href = `/api/scans/${currentScan.id}/export-csv`;
});

refreshHistoryBtn.addEventListener("click", loadHistory);

async function runAnalysis(endpoint, formData) {
  setStatus("Processing DIP pipeline...", true);
  try {
    const res = await fetch(endpoint, { method: "POST", body: formData });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Analysis failed");
    }
    const data = await res.json();
    renderScan(data.scan);
    await loadHistory();
    setStatus("Analysis Complete", false);
  } catch (err) {
    console.error(err);
    setStatus(`Error: ${err.message}`, false);
  }
}

function setStatus(text, busy) {
  statusPill.textContent = text;
  uploadBtn.disabled = busy;
  sampleBtn.disabled = busy;
}

function renderScan(scan) {
  currentScan = scan;
  exportCsvBtn.disabled = false;

  imgRaw.src = showEnhancedToggle.checked ? scan.enhanced_url : scan.raw_url;
  imgHeatmap.src = scan.heatmap_url;
  imgMask.src = scan.mask_url;

  [imgRaw, imgHeatmap, imgMask].forEach((el) => (el.style.display = "block"));
  [phRaw, phHeatmap, phMask].forEach((el) => (el.style.display = "none"));

  const healthyPct = Math.max(
    0,
    scan.canopy_coverage_pct - scan.stressed_area_pct
  ).toFixed(2);

  document.getElementById("valCanopy").textContent = `${scan.canopy_coverage_pct}%`;
  document.getElementById("valStressed").textContent = `${scan.stressed_area_pct}%`;
  document.getElementById("valSoil").textContent = `${scan.bare_soil_pct}%`;

  document.getElementById("barHealthy").style.width = `${healthyPct}%`;
  document.getElementById("pctHealthy").textContent = `${healthyPct}%`;

  document.getElementById("barStressed").style.width = `${scan.stressed_area_pct}%`;
  document.getElementById("pctStressed").textContent = `${scan.stressed_area_pct}%`;

  document.getElementById("barSoil").style.width = `${scan.bare_soil_pct}%`;
  document.getElementById("pctSoil").textContent = `${scan.bare_soil_pct}%`;

  document.getElementById("valOtsu").textContent = scan.otsu_threshold;
  document.getElementById("valIndices").textContent = `${scan.mean_exg} / ${scan.mean_vari}`;
  document.getElementById("valClusters").textContent = scan.cluster_count;
  document.getElementById("valCompactness").textContent = scan.avg_compactness;
  document.getElementById("valUniformity").textContent = `${scan.row_uniformity_pct}%`;
}

async function loadHistory() {
  try {
    const res = await fetch("/api/scans");
    const data = await res.json();
    const tbody = document.getElementById("historyBody");
    if (!data.scans || data.scans.length === 0) {
      tbody.innerHTML = '<tr><td colspan="9" class="empty-row">No scans recorded yet.</td></tr>';
      return;
    }

    tbody.innerHTML = "";
    data.scans.forEach((scan) => {
      const tr = document.createElement("tr");
      const values = [
        `#${scan.id}`,
        scan.plot_name,
        scan.filename,
        scan.gamma,
        `${scan.canopy_coverage_pct}%`,
        `${scan.stressed_area_pct}%`,
        `${scan.row_uniformity_pct}%`,
        scan.cluster_count,
      ];
      values.forEach((value, index) => {
        const td = document.createElement("td");
        td.textContent = String(value);
        if (index === 4) td.style.fontWeight = "700";
        tr.appendChild(td);
      });

      const actionTd = document.createElement("td");
      const viewButton = document.createElement("button");
      viewButton.className = "btn btn-small view-btn";
      viewButton.textContent = "View";
      viewButton.addEventListener("click", () => renderScan(scan));

      const deleteButton = document.createElement("button");
      deleteButton.className = "btn btn-small btn-danger del-btn";
      deleteButton.textContent = "Delete";
      deleteButton.addEventListener("click", async () => {
        await fetch(`/api/scans/${scan.id}`, { method: "DELETE" });
        await loadHistory();
      });

      actionTd.append(viewButton, deleteButton);
      tr.appendChild(actionTd);
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.error("Failed to load history:", err);
  }
}

loadHistory();
