/**
 * main.js — VISOR  Object Detection Dashboard
 * Handles: tabs, settings, ROI canvas, analytics polling,
 *          START / PAUSE / STOP, VIDEO SEEK BAR
 */

// ── State ──────────────────────────────────────────────────────────────────
const state = {
  selectedClasses   : [],
  roiZones          : [],
  drawingMode       : false,
  pendingZoneName   : "",
  analyticsInterval : null,
  progressInterval  : null,
  currentState      : "stopped",
  isFilePlaying     : false,   // true when a video FILE (not webcam) is active
  isSeeking         : false,   // true while user is dragging the scrubber
};

// ── DOM Helpers ──────────────────────────────────────────────────────────────
const $  = id  => document.getElementById(id);
const $$ = sel => document.querySelectorAll(sel);

// ── Tab Switching ────────────────────────────────────────────────────────────
$$(".tab-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    const target = btn.dataset.tab;
    $$(".tab-btn").forEach(b => b.classList.remove("active"));
    $$(".tab-panel").forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
    $(`panel-${target}`).classList.add("active");
    if (target === "heatmap") activateHeatmap();
  });
});

// ── Range Sliders ─────────────────────────────────────────────────────────────
function bindRange(inputId, displayId, decimals = 2) {
  const input = $(inputId), display = $(displayId);
  if (!input || !display) return;
  const update = () => { display.textContent = parseFloat(input.value).toFixed(decimals); };
  input.addEventListener("input", update);
  update();
}
bindRange("confRange", "confVal", 2);
bindRange("iouRange",  "iouVal",  2);

// ── Class Filter ──────────────────────────────────────────────────────────────
function buildClassGrid(classes) {
  const grid = $("classGrid");
  if (!grid) return;
  grid.innerHTML = "";
  classes.forEach(cls => {
    const div = document.createElement("div");
    div.className = "class-chip"; div.dataset.cls = cls;
    div.innerHTML = `<input type="checkbox" id="cls-${cls}" value="${cls}">
      <label for="cls-${cls}" style="margin:0;color:inherit;cursor:pointer">${cls}</label>`;
    div.querySelector("input").addEventListener("change", e => {
      div.classList.toggle("selected", e.target.checked); syncClassFilter();
    });
    grid.appendChild(div);
  });
}
function syncClassFilter() {
  state.selectedClasses = Array.from($$("#classGrid input:checked")).map(cb => cb.value);
  applySettings({ classes: state.selectedClasses });
}
$("classSearch") && $("classSearch").addEventListener("input", () => {
  const q = $("classSearch").value.toLowerCase();
  $$(".class-chip").forEach(c => { c.style.display = c.dataset.cls.includes(q) ? "" : "none"; });
});
$("selectAllClasses") && $("selectAllClasses").addEventListener("click", () => {
  $$("#classGrid input").forEach(cb => { cb.checked = true; cb.closest(".class-chip").classList.add("selected"); });
  syncClassFilter();
});
$("clearAllClasses") && $("clearAllClasses").addEventListener("click", () => {
  $$("#classGrid input").forEach(cb => { cb.checked = false; cb.closest(".class-chip").classList.remove("selected"); });
  state.selectedClasses = []; applySettings({ classes: [] });
});

// ── Settings API ──────────────────────────────────────────────────────────────
async function applySettings(overrides = {}) {
  const payload = {
    model        : $("modelSelect")    ? $("modelSelect").value           : undefined,
    confidence   : $("confRange")      ? parseFloat($("confRange").value) : undefined,
    iou          : $("iouRange")       ? parseFloat($("iouRange").value)  : undefined,
    show_heatmap : $("toggleHeatmap")  ? $("toggleHeatmap").checked       : undefined,
    show_trails  : $("toggleTrails")   ? $("toggleTrails").checked        : undefined,
    tracking_on  : $("toggleTracking") ? $("toggleTracking").checked      : undefined,
    show_roi     : $("toggleROI")      ? $("toggleROI").checked           : undefined,
    classes      : state.selectedClasses,
    ...overrides,
  };
  Object.keys(payload).forEach(k => payload[k] === undefined && delete payload[k]);
  try {
    await fetch("/api/settings", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  } catch (e) { console.error("Settings error:", e); }
}
["modelSelect","confRange","iouRange","toggleHeatmap","toggleTrails","toggleTracking","toggleROI"]
  .forEach(id => { const el = $(id); if (el) el.addEventListener("change", () => applySettings()); });

// ── Video Source ──────────────────────────────────────────────────────────────
$("sourceSelect") && $("sourceSelect").addEventListener("change", () => {
  const val = $("sourceSelect").value;
  if ($("fileRow")) $("fileRow").style.display = val === "file" ? "block" : "none";
  if (val !== "file") {
    state.isFilePlaying = false;
    updateSeekBarVisibility(false);
    applySettings({ source: val });
  }
});

$("videoFileInput") && $("videoFileInput").addEventListener("change", async () => {
  const file = $("videoFileInput").files[0];
  if (!file) return;

  // Show upload progress
  showUploadStatus("uploading", file.name);

  const formData = new FormData();
  formData.append("video", file);
  try {
    const res  = await fetch("/api/upload_video", { method: "POST", body: formData });
    const data = await res.json();
    if (data.path) {
      showUploadStatus("success", file.name);
      state.isFilePlaying = true;
      applySettings({ source: data.path });
    } else {
      showUploadStatus("error", data.error || "Unknown error");
    }
  } catch (e) {
    showUploadStatus("error", "Upload failed — is the server running?");
  }
});

function showUploadStatus(type, message) {
  const el = $("uploadStatus");
  if (!el) return;
  const styles = {
    uploading: "color:var(--accent-amber)",
    success  : "color:var(--accent-green)",
    error    : "color:var(--accent-red)",
  };
  const icons = { uploading: "⏳", success: "✅", error: "❌" };
  el.style.cssText = `font-size:11px;margin-top:4px;${styles[type] || ""}`;
  el.textContent   = `${icons[type] || ""} ${message}`;
  el.style.display = "block";
}

// =============================================================================
//  START / PAUSE / STOP Controls
// =============================================================================

function updateControlUI(newState) {
  state.currentState = newState;

  const btnStart = $("btnStart"), btnPause = $("btnPause"),
        btnStop  = $("btnStop"),  badge    = $("stateBadge"),
        dot      = $("statusDot"),dotLabel = $("statusLabel");

  if (!btnStart) return;

  if (newState === "stopped") {
    btnStart.disabled = false; btnStart.textContent = "▶  Start";
    btnStart.className = "btn btn-start";
    btnPause.disabled = true;  btnPause.textContent = "⏸  Pause";
    btnPause.className = "btn btn-pause btn-disabled";
    btnStop.disabled  = true;  btnStop.className = "btn btn-stop btn-disabled";
    stopProgressPolling();

  } else if (newState === "running") {
    btnStart.disabled = true;  btnStart.textContent = "▶  Running…";
    btnStart.className = "btn btn-start btn-disabled";
    btnPause.disabled = false; btnPause.textContent = "⏸  Pause";
    btnPause.className = "btn btn-pause";
    btnStop.disabled  = false; btnStop.className = "btn btn-stop";
    startProgressPolling();

  } else if (newState === "paused") {
    btnStart.disabled = true;  btnStart.textContent = "▶  Running…";
    btnStart.className = "btn btn-start btn-disabled";
    btnPause.disabled = false; btnPause.textContent = "▶  Resume";
    btnPause.className = "btn btn-resume";
    btnStop.disabled  = false; btnStop.className = "btn btn-stop";
  }

  if (badge)    { badge.textContent = newState.toUpperCase(); badge.className = `state-badge state-${newState}`; }
  if (dot)      { dot.className = `status-dot dot-${newState}`; }
  if (dotLabel) { dotLabel.textContent = { stopped:"STOPPED", running:"LIVE", paused:"PAUSED" }[newState] || newState.toUpperCase(); }
}

$("btnStart") && $("btnStart").addEventListener("click", async () => {
  try {
    const ep   = state.currentState === "paused" ? "/api/resume" : "/api/start";
    const data = await (await fetch(ep, { method: "POST" })).json();
    updateControlUI(data.state);
  } catch (e) { console.error("Start error:", e); }
});

$("btnPause") && $("btnPause").addEventListener("click", async () => {
  try {
    const ep   = state.currentState === "paused" ? "/api/resume" : "/api/pause";
    const data = await (await fetch(ep, { method: "POST" })).json();
    updateControlUI(data.state);
  } catch (e) { console.error("Pause error:", e); }
});

$("btnStop") && $("btnStop").addEventListener("click", async () => {
  try {
    const data = await (await fetch("/api/stop", { method: "POST" })).json();
    updateControlUI(data.state);
    resetSeekBar();
  } catch (e) { console.error("Stop error:", e); }
});

// =============================================================================
//  SEEK BAR
// =============================================================================

const seekBar       = $("seekBar");
const seekCurrent   = $("seekCurrent");
const seekTotal     = $("seekTotal");
const seekContainer = $("seekContainer");

/** Show or hide the entire seek bar row */
function updateSeekBarVisibility(visible) {
  if (seekContainer) {
    seekContainer.style.display = visible ? "flex" : "none";
  }
}

/** Reset seek bar to 0 */
function resetSeekBar() {
  if (seekBar)     { seekBar.value = 0; }
  if (seekCurrent) { seekCurrent.textContent = "00:00"; }
  if (seekTotal)   { seekTotal.textContent   = "00:00"; }
  updateSeekBarVisibility(false);
}

/** Update the seek bar position from a progress dict */
function updateSeekBar(progress) {
  if (!progress.is_file) {
    updateSeekBarVisibility(false);
    return;
  }

  updateSeekBarVisibility(true);

  // Don't overwrite value while user is dragging
  if (!state.isSeeking && seekBar) {
    seekBar.value = Math.round(progress.fraction * 1000);
  }
  if (seekCurrent) seekCurrent.textContent = progress.current_time || "00:00";
  if (seekTotal)   seekTotal.textContent   = progress.total_time   || "00:00";
}

// ── Seek bar input events ─────────────────────────────────────────────────────

if (seekBar) {
  // While dragging — update time display but don't send request yet
  seekBar.addEventListener("input", () => {
    state.isSeeking = true;
    const fraction  = seekBar.value / 1000;

    // Update time display optimistically
    if (seekTotal && seekTotal.textContent !== "00:00") {
      const totalParts = seekTotal.textContent.split(":").map(Number);
      const totalSec   = totalParts[0] * 60 + totalParts[1];
      const curSec     = Math.round(fraction * totalSec);
      const m = Math.floor(curSec / 60), s = curSec % 60;
      if (seekCurrent) seekCurrent.textContent = `${String(m).padStart(2,"0")}:${String(s).padStart(2,"0")}`;
    }
  });

  // On mouse/touch release — send the seek request to server
  seekBar.addEventListener("change", async () => {
    state.isSeeking = false;
    const fraction  = seekBar.value / 1000;
    try {
      await fetch("/api/seek", {
        method : "POST",
        headers: { "Content-Type": "application/json" },
        body   : JSON.stringify({ position: fraction }),
      });
    } catch (e) { console.error("Seek error:", e); }
  });
}

// ── Progress polling ──────────────────────────────────────────────────────────

function startProgressPolling() {
  if (state.progressInterval) return;
  state.progressInterval = setInterval(fetchProgress, 500);
}

function stopProgressPolling() {
  if (state.progressInterval) {
    clearInterval(state.progressInterval);
    state.progressInterval = null;
  }
}

async function fetchProgress() {
  try {
    const res      = await fetch("/api/video_progress");
    const progress = await res.json();
    updateSeekBar(progress);
  } catch (e) { /* ignore */ }
}

// =============================================================================
//  Analytics Polling
// =============================================================================

function startAnalyticsPolling() {
  if (state.analyticsInterval) clearInterval(state.analyticsInterval);
  state.analyticsInterval = setInterval(fetchAnalytics, 700);
}

async function fetchAnalytics() {
  try {
    const res  = await fetch("/api/analytics");
    const data = await res.json();
    updateDashboard(data);
    if (data.state && data.state !== state.currentState) updateControlUI(data.state);
  } catch (e) { /* server not ready */ }
}

function updateDashboard(data) {
  $("statFPS")    && ($("statFPS").textContent    = (data.fps || 0).toFixed(1));
  $("statCount")  && ($("statCount").textContent  = data.current_count || 0);
  $("statTotal")  && ($("statTotal").textContent  = data.total_tracked || 0);
  $("statFrames") && ($("statFrames").textContent = (data.frame_count || 0).toLocaleString());
  $("sbFPS")    && ($("sbFPS").textContent    = (data.fps || 0).toFixed(1));
  $("sbLive")   && ($("sbLive").textContent   = data.current_count || 0);
  $("sbTotal")  && ($("sbTotal").textContent  = data.total_tracked || 0);
  $("sbModel")  && ($("sbModel").textContent  = (data.model || "").replace(".pt","").toUpperCase());
  $("sbSource") && ($("sbSource").textContent = data.source || "—");
  $("sbUptime") && ($("sbUptime").textContent = data.uptime || "00:00:00");

  const table = $("classCountBody");
  if (table && data.class_counts_now) {
    const rows = Object.entries(data.class_counts_now).sort((a,b)=>b[1]-a[1]).slice(0,15);
    table.innerHTML = rows.map(([cls,cnt])=>`<tr><td>${cls}</td><td>${cnt}</td></tr>`).join("")
      || `<tr><td colspan="2" style="color:var(--text-muted);text-align:center">No detections</td></tr>`;
  }

  const alertList = $("roiAlertList");
  if (alertList && data.roi_alerts) {
    alertList.innerHTML = data.roi_alerts.length === 0
      ? `<div style="color:var(--text-muted);font-size:11px;text-align:center;padding:8px">No zones defined</div>`
      : data.roi_alerts.map(a =>
          `<div class="alert-item ${a.count>0?'active':''}">
            <span>${a.zone}</span>
            <span class="alert-count ${a.count>0?'danger':''}">${a.count}</span>
          </div>`).join("");
  }
}

// =============================================================================
//  ROI Canvas
// =============================================================================

const canvas = $("roiCanvas"), drawBadge = $("drawModeBadge");
let isDrawing = false, startX, startY;

function enterDrawMode() {
  if (!canvas) return;
  const name = prompt("Zone name:", `Zone ${state.roiZones.length + 1}`);
  if (!name) return;
  state.pendingZoneName = name; state.drawingMode = true;
  canvas.style.pointerEvents = "all";
  if (drawBadge) drawBadge.classList.add("visible");
}
function exitDrawMode() {
  state.drawingMode = false;
  if (canvas) canvas.style.pointerEvents = "none";
  if (drawBadge) drawBadge.classList.remove("visible");
  redrawROI();
}
function getCanvasPos(e) {
  const r = canvas.getBoundingClientRect();
  const cx = e.touches ? e.touches[0].clientX : e.clientX;
  const cy = e.touches ? e.touches[0].clientY : e.clientY;
  return { x: Math.round((cx-r.left)*canvas.width/r.width), y: Math.round((cy-r.top)*canvas.height/r.height) };
}
if (canvas) {
  const resize = () => { const img=$("videoFeed"); if(img){canvas.width=img.clientWidth||640;canvas.height=img.clientHeight||480;} };
  window.addEventListener("resize", resize); setTimeout(resize, 500);
  canvas.addEventListener("mousedown", e => { if(!state.drawingMode)return; isDrawing=true; const p=getCanvasPos(e); startX=p.x; startY=p.y; });
  canvas.addEventListener("mousemove", e => {
    if(!isDrawing||!state.drawingMode)return;
    const p=getCanvasPos(e), ctx=canvas.getContext("2d");
    ctx.clearRect(0,0,canvas.width,canvas.height); redrawROI(ctx);
    ctx.strokeStyle="#ffb800"; ctx.lineWidth=2; ctx.setLineDash([6,3]);
    ctx.strokeRect(startX,startY,p.x-startX,p.y-startY);
    ctx.fillStyle="rgba(255,184,0,0.08)"; ctx.fillRect(startX,startY,p.x-startX,p.y-startY);
    ctx.setLineDash([]);
  });
  canvas.addEventListener("mouseup", e => {
    if(!isDrawing||!state.drawingMode)return; isDrawing=false;
    const p=getCanvasPos(e);
    const x1=Math.min(startX,p.x),y1=Math.min(startY,p.y),x2=Math.max(startX,p.x),y2=Math.max(startY,p.y);
    if(Math.abs(x2-x1)>15&&Math.abs(y2-y1)>15){
      const img=$("videoFeed");
      const sx=(img&&img.naturalWidth)?img.naturalWidth/canvas.width:1;
      const sy=(img&&img.naturalHeight)?img.naturalHeight/canvas.height:1;
      state.roiZones.push({name:state.pendingZoneName,x1:Math.round(x1*sx),y1:Math.round(y1*sy),x2:Math.round(x2*sx),y2:Math.round(y2*sy)});
      sendROI();
    }
    exitDrawMode();
  });
}
function redrawROI(ctx) {
  if(!canvas)return; ctx=ctx||canvas.getContext("2d"); ctx.clearRect(0,0,canvas.width,canvas.height);
  const colors=["#ffb800","#00ff88","#ff3366","#a855f7","#00d4ff"];
  state.roiZones.forEach((z,i)=>{
    const img=$("videoFeed");
    const sx=(img&&img.naturalWidth)?canvas.width/img.naturalWidth:1;
    const sy=(img&&img.naturalHeight)?canvas.height/img.naturalHeight:1;
    const color=colors[i%colors.length];
    ctx.strokeStyle=color; ctx.lineWidth=2;
    ctx.strokeRect(z.x1*sx,z.y1*sy,(z.x2-z.x1)*sx,(z.y2-z.y1)*sy);
    ctx.fillStyle=color+"22"; ctx.fillRect(z.x1*sx,z.y1*sy,(z.x2-z.x1)*sx,(z.y2-z.y1)*sy);
    ctx.fillStyle=color; ctx.font="bold 11px 'JetBrains Mono',monospace";
    ctx.fillText(z.name,z.x1*sx+4,z.y1*sy+14);
  });
}
async function sendROI() {
  try { await fetch("/api/roi",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({zones:state.roiZones})}); renderZoneList(); }
  catch(e){console.error("ROI error:",e);}
}
function renderZoneList() {
  const list=$("roiZoneList"); if(!list)return;
  const colors=["#ffb800","#00ff88","#ff3366","#a855f7","#00d4ff"];
  list.innerHTML=state.roiZones.length===0
    ?`<div style="color:var(--text-muted);font-size:11px;text-align:center;padding:8px">No zones yet</div>`
    :state.roiZones.map((z,i)=>`<div class="roi-zone-item">
        <div style="display:flex;align-items:center;gap:6px">
          <div class="roi-zone-dot" style="background:${colors[i%colors.length]}"></div>
          <span>${z.name}</span>
        </div>
        <button class="roi-del-btn" onclick="deleteZone(${i})">✕</button>
      </div>`).join("");
  redrawROI();
}
function deleteZone(idx){state.roiZones.splice(idx,1);sendROI();renderZoneList();redrawROI();}
window.deleteZone=deleteZone;
$("btnDrawROI")  && $("btnDrawROI").addEventListener("click", enterDrawMode);
$("btnClearROI") && $("btnClearROI").addEventListener("click",()=>{state.roiZones=[];sendROI();renderZoneList();});

// ── Heatmap ───────────────────────────────────────────────────────────────────
function activateHeatmap(){const img=$("heatmapImg");if(img&&!img.src.includes("/heatmap_feed"))img.src="/heatmap_feed";}
$("btnClearHeatmap")&&$("btnClearHeatmap").addEventListener("click",async()=>{await fetch("/api/clear_heatmap",{method:"POST"});});

// ── Other controls ────────────────────────────────────────────────────────────
$("btnReset")&&$("btnReset").addEventListener("click",async()=>{
  if(!confirm("Reset tracker and analytics?"))return;
  await fetch("/api/reset",{method:"POST"});
});
$("btnDownloadCSV")&&$("btnDownloadCSV").addEventListener("click",()=>{window.location.href="/api/download_csv";});
$("btnSnapshot")&&$("btnSnapshot").addEventListener("click",()=>{
  const img=$("videoFeed"); if(!img)return;
  const tmp=document.createElement("canvas");
  tmp.width=img.naturalWidth||640; tmp.height=img.naturalHeight||480;
  tmp.getContext("2d").drawImage(img,0,0);
  const link=document.createElement("a"); link.download=`snapshot_${Date.now()}.jpg`;
  link.href=tmp.toDataURL("image/jpeg",0.92); link.click();
});

// =============================================================================
//  Init
// =============================================================================
document.addEventListener("DOMContentLoaded", () => {
  fetch("/api/classes").then(r=>r.json()).then(buildClassGrid).catch(()=>{});
  fetch("/api/state").then(r=>r.json()).then(d=>updateControlUI(d.state||"stopped")).catch(()=>updateControlUI("stopped"));
  startAnalyticsPolling();
  resetSeekBar();
  if(canvas) canvas.style.pointerEvents="none";
  fetchAnalytics();
});