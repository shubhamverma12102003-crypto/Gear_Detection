import streamlit as st
import requests
import os
import tempfile
import time
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
from collections import Counter
from ultralytics import YOLO

st.set_page_config(
    page_title="GearVision AI",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;400;500;600;700&family=Share+Tech+Mono&family=Exo+2:wght@200;300;400;500;600;700;800&display=swap');

:root {
    --bg:          #07090d;
    --bg2:         #0d1117;
    --surface:     #111820;
    --panel:       #161e28;
    --panel2:      #1c2535;
    --border:      #1e2d3d;
    --border2:     #253446;
    --gold:        #d4a843;
    --gold2:       #f0c060;
    --gold-dim:    #8a6820;
    --steel:       #8ab4d4;
    --steel2:      #b0ccdf;
    --cyan:        #00d4ff;
    --red:         #ff4444;
    --green:       #00e676;
    --yellow:      #ffd740;
    --text:        #cdd6e0;
    --text2:       #8a9bb0;
    --text3:       #4a5a6a;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text);
    font-family: 'Exo 2', sans-serif;
}
[data-testid="stSidebar"] {
    background: var(--bg2) !important;
    border-right: 1px solid var(--border) !important;
}
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
[data-testid="stDecoration"] { display: none; }

.hero-wrap {
    position: relative;
    padding: 2.5rem 1rem 2rem;
    text-align: center;
    overflow: hidden;
}
.hero-bg-lines {
    position: absolute;
    inset: 0;
    background-image:
        repeating-linear-gradient(90deg, transparent, transparent 59px, rgba(212,168,67,0.06) 59px, rgba(212,168,67,0.06) 60px),
        repeating-linear-gradient(0deg,  transparent, transparent 59px, rgba(212,168,67,0.04) 59px, rgba(212,168,67,0.04) 60px);
    pointer-events: none;
}
.hero-tag {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.35em;
    color: var(--gold-dim);
    text-transform: uppercase;
    margin-bottom: 0.6rem;
    display: block;
}
.hero-title {
    font-family: 'Rajdhani', sans-serif;
    font-weight: 700;
    font-size: clamp(2.8rem, 6vw, 4.8rem);
    line-height: 0.95;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--text);
    position: relative;
}
.hero-title .gear   { color: var(--gold); }
.hero-title .vision { color: var(--steel2); }
.hero-title .ai     { color: var(--text3); }
.hero-sub {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.7rem;
    color: var(--text3);
    letter-spacing: 0.22em;
    text-transform: uppercase;
    margin-top: 0.8rem;
}
.hero-badge-row {
    display: flex;
    justify-content: center;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin-top: 1.2rem;
}
.hero-badge {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.6rem;
    padding: 0.28rem 0.7rem;
    border: 1px solid var(--border2);
    color: var(--text2);
    letter-spacing: 0.1em;
    background: var(--surface);
    clip-path: polygon(6px 0%, 100% 0%, calc(100% - 6px) 100%, 0% 100%);
}
.sep-line {
    height: 1px;
    background: linear-gradient(90deg, transparent 0%, var(--gold-dim) 25%, var(--gold) 50%, var(--gold-dim) 75%, transparent 100%);
    opacity: 0.45;
}

/* CARDS */
.card {
    background: var(--panel);
    border: 1px solid var(--border);
    clip-path: polygon(0 0, calc(100% - 18px) 0, 100% 18px, 100% 100%, 18px 100%, 0 calc(100% - 18px));
    padding: 1.5rem;
    position: relative;
    margin: 0.6rem 0;
}
.card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--gold), transparent 60%);
}
.card-danger::after { background: linear-gradient(90deg, var(--red),    transparent 60%); }
.card-warn::after   { background: linear-gradient(90deg, var(--yellow),  transparent 60%); }
.card-ok::after     { background: linear-gradient(90deg, var(--green),   transparent 60%); }
.card-steel::after  { background: linear-gradient(90deg, var(--steel),   transparent 60%); }

/* METRICS */
.metric-box {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-top: 2px solid var(--gold-dim);
    padding: 1rem 0.75rem;
    text-align: center;
    clip-path: polygon(0 0, calc(100% - 10px) 0, 100% 10px, 100% 100%, 0 100%);
}
.metric-val {
    font-family: 'Rajdhani', sans-serif;
    font-weight: 700;
    font-size: 2rem;
    color: var(--gold2);
    line-height: 1;
}
.metric-lbl {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.58rem;
    color: var(--text3);
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-top: 0.3rem;
}

/* VERDICT */
.verdict-box {
    text-align: center;
    padding: 1.5rem 1rem;
    background: var(--panel);
    border: 1px solid var(--border);
    clip-path: polygon(0 0, calc(100% - 20px) 0, 100% 20px, 100% 100%, 20px 100%, 0 calc(100% - 20px));
}
.verdict-label {
    font-family: 'Rajdhani', sans-serif;
    font-weight: 700;
    font-size: 1.9rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
}
.verdict-bar {
    height: 3px;
    margin: 0.75rem auto;
    width: 60%;
    opacity: 0.5;
}
.verdict-action {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.65rem;
    color: var(--text2);
    letter-spacing: 0.1em;
}

/* BADGES */
.badge {
    display: inline-block;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.67rem;
    padding: 0.22rem 0.65rem;
    letter-spacing: 0.06em;
}
.b-break   { background: rgba(255,68,68,0.12); border: 1px solid rgba(255,68,68,0.45); color: #ff7070; }
.b-lack    { background: rgba(0,230,118,0.1);  border: 1px solid rgba(0,230,118,0.35); color: #50ffaa; }
.b-scratch { background: rgba(0,212,255,0.1);  border: 1px solid rgba(0,212,255,0.35); color: #60ddff; }

/* CONF BAR */
.cbar-wrap { background: var(--bg2); height: 4px; margin-top: 5px; }
.cbar      { height: 4px; }

/* SECTION HEADER */
.sec-hdr {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.6rem;
    color: var(--text3);
    letter-spacing: 0.25em;
    text-transform: uppercase;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.sec-hdr::before {
    content: '';
    display: block;
    width: 14px;
    height: 1px;
    background: var(--gold);
    flex-shrink: 0;
}

/* SIDEBAR */
.sb-logo {
    font-family: 'Rajdhani', sans-serif;
    font-weight: 700;
    font-size: 1.25rem;
    letter-spacing: 0.12em;
    color: var(--gold);
    text-align: center;
    text-transform: uppercase;
    padding: 0.5rem 0;
}
.sb-logo span { color: var(--steel2); }
.sb-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border2), transparent);
    margin: 0.75rem 0;
}
.sb-stat {
    background: var(--panel);
    border-left: 2px solid var(--gold-dim);
    padding: 0.45rem 0.75rem;
    margin: 0.28rem 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.65rem;
}
.sb-stat-key { color: var(--text2); }
.sb-stat-val { color: var(--gold2); }

/* UPLOAD */
[data-testid="stFileUploader"] {
    border: 1px dashed var(--border2) !important;
    border-radius: 0 !important;
    background: var(--panel) !important;
}
[data-testid="stFileUploader"]:hover { border-color: var(--gold) !important; }

/* BUTTONS */
.stButton > button {
    background: transparent !important;
    color: var(--gold) !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.2em !important;
    border: 1px solid var(--gold-dim) !important;
    border-radius: 0 !important;
    padding: 0.55rem 1.5rem !important;
    text-transform: uppercase !important;
    width: 100%;
}
.stButton > button:hover {
    background: rgba(212,168,67,0.1) !important;
    border-color: var(--gold) !important;
}

/* MISC */
p, li, label { color: var(--text) !important; font-family: 'Exo 2', sans-serif !important; }
h1, h2, h3   { font-family: 'Rajdhani', sans-serif !important; color: var(--text) !important; }
[data-baseweb="select"] > div { background: var(--panel) !important; border-color: var(--border2) !important; border-radius: 0 !important; }

.scan-text {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.8rem;
    color: var(--gold);
    letter-spacing: 0.3em;
    text-align: center;
    animation: blink 0.8s step-end infinite;
}
@keyframes blink { 50% { opacity: 0.15; } }

.img-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.58rem;
    color: var(--text3);
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.det-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.38rem 0;
    border-bottom: 1px solid var(--border);
}
.det-idx  { font-family: 'Share Tech Mono', monospace; font-size: 0.62rem; color: var(--text3); min-width: 24px; }
.det-conf { font-family: 'Rajdhani', sans-serif; font-weight: 700; font-size: 0.85rem; }
.defect-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin: 0.55rem 0;
}
.defect-count { font-family: 'Share Tech Mono', monospace; font-size: 0.68rem; color: var(--text2); }
.defect-avg   { font-family: 'Rajdhani', sans-serif; font-weight: 600; font-size: 0.85rem; margin-left: auto; color: var(--text); }

.defect-info-row {
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
    padding: 0.55rem 0;
    border-bottom: 1px solid var(--border);
}
.defect-dot {
    width: 8px; height: 8px;
    clip-path: polygon(50% 0%, 100% 50%, 50% 100%, 0% 50%);
    margin-top: 4px;
    flex-shrink: 0;
}
.defect-info-name { font-family: 'Rajdhani', sans-serif; font-weight: 600; font-size: 0.85rem; color: var(--text); text-transform: uppercase; letter-spacing: 0.05em; }
.defect-info-desc { font-family: 'Share Tech Mono', monospace; font-size: 0.6rem; color: var(--text3); margin-top: 2px; }

.empty-state {
    text-align: center;
    padding: 4rem 2rem;
    background: var(--panel);
    border: 1px dashed var(--border2);
    clip-path: polygon(0 0, calc(100% - 24px) 0, 100% 24px, 100% 100%, 24px 100%, 0 calc(100% - 24px));
}
.empty-icon  { font-family: 'Rajdhani',sans-serif; font-size: 4rem; color: var(--border2); line-height:1; margin-bottom:1rem; }
.empty-title { font-family: 'Rajdhani',sans-serif; font-weight:600; font-size:1.1rem; letter-spacing:0.2em; color:var(--text3); text-transform:uppercase; margin-bottom:0.5rem; }
.empty-sub   { font-family: 'Share Tech Mono',monospace; font-size:0.63rem; color:var(--text3); }
.no-defect-msg { font-family:'Rajdhani',sans-serif; font-size:1.05rem; font-weight:600; color:var(--green); letter-spacing:0.05em; }

/* ── DEVELOPER FOOTER ── */
.footer-outer {
    margin-top: 2rem;
    padding: 2rem 1rem 1.5rem;
    border-top: 1px solid var(--border);
}
.footer-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.58rem;
    color: var(--text3);
    letter-spacing: 0.35em;
    text-transform: uppercase;
    text-align: center;
    margin-bottom: 1.2rem;
}
.dev-row {
    display: flex;
    justify-content: center;
    gap: 1rem;
    flex-wrap: wrap;
    margin-bottom: 1.2rem;
}
.dev-card {
    background: var(--panel);
    border: 1px solid var(--border2);
    clip-path: polygon(0 0, calc(100% - 14px) 0, 100% 14px, 100% 100%, 14px 100%, 0 calc(100% - 14px));
    padding: 1rem 1.5rem;
    display: flex;
    align-items: center;
    gap: 0.85rem;
    min-width: 210px;
}
.dev-avatar {
    width: 42px; height: 42px;
    clip-path: polygon(0 0, calc(100% - 8px) 0, 100% 8px, 100% 100%, 8px 100%, 0 calc(100% - 8px));
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'Rajdhani', sans-serif;
    font-weight: 700;
    font-size: 0.9rem;
    flex-shrink: 0;
}
.av-gold  { background: rgba(212,168,67,0.15); border: 1px solid var(--gold-dim); color: var(--gold); }
.av-steel { background: rgba(138,180,212,0.15); border: 1px solid var(--steel); color: var(--steel2); }
.dev-name { font-family: 'Rajdhani', sans-serif; font-weight: 600; font-size: 0.95rem; color: var(--text); letter-spacing: 0.05em; text-transform: uppercase; }
.dev-role { font-family: 'Share Tech Mono', monospace; font-size: 0.56rem; color: var(--gold-dim); letter-spacing: 0.12em; margin-top: 2px; }
.footer-copy { font-family: 'Share Tech Mono', monospace; font-size: 0.56rem; color: var(--text3); letter-spacing: 0.1em; text-align: center; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
MODEL_URL  = "https://huggingface.co/Shubhamv12/Gear_Detection/resolve/main/best.pt"
MODEL_PATH = Path("best.pt")
CLASS_NAMES  = {0: "break", 1: "lack", 2: "scratch"}
CLASS_COLORS = {
    "break"  : (255, 68,  68),
    "lack"   : (0,   230, 118),
    "scratch": (0,   212, 255),
}

@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_PATH.exists():
        with requests.get(MODEL_URL, stream=True) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            downloaded = 0
            bar = st.progress(0, text="Acquiring model from HuggingFace...")
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        bar.progress(downloaded / total,
                                     text=f"Downloading… {downloaded/1e6:.1f} / {total/1e6:.1f} MB")
            bar.empty()
    return YOLO(str(MODEL_PATH))

def run_inference(model, img_path, conf, iou, imgsz, augment):
    results = model.predict(source=str(img_path), imgsz=imgsz, conf=conf,
                            iou=iou, device="cpu", augment=augment, verbose=False)
    return results[0]

def draw_detections(result, img_bgr):
    drawn = img_bgr.copy()
    detections = []
    if result.obb is not None and len(result.obb) > 0:
        for i in range(len(result.obb)):
            cls_id   = int(result.obb.cls[i].item())
            conf_val = float(result.obb.conf[i].item())
            cls_name = CLASS_NAMES.get(cls_id, "unknown")
            color    = CLASS_COLORS.get(cls_name, (200, 200, 200))
            pts = result.obb.xyxyxyxy[i].cpu().numpy().reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(drawn, [pts], isClosed=True, color=color, thickness=3)
            x1 = int(result.obb.xyxyxyxy[i].cpu().numpy()[:, 0].min())
            y1 = int(result.obb.xyxyxyxy[i].cpu().numpy()[:, 1].min())
            label = f"{cls_name} {conf_val:.0%}"
            font  = cv2.FONT_HERSHEY_SIMPLEX
            fs, th = 0.6, 2
            (tw, thh), _ = cv2.getTextSize(label, font, fs, th)
            cv2.rectangle(drawn, (x1, y1 - thh - 10), (x1 + tw + 6, y1), color, -1)
            cv2.putText(drawn, label, (x1 + 3, y1 - 4), font, fs, (0, 0, 0), th)
            detections.append({"class": cls_name, "conf": conf_val})
    return drawn, detections

def get_severity(total, detections):
    if total == 0:
        return "NORMAL", "normal", "✅"
    classes = set(d["class"] for d in detections)
    if total >= 5 or "break" in classes:
        return "HIGH",   "high",   "🔴"
    elif total >= 2:
        return "MEDIUM", "medium", "🟡"
    else:
        return "LOW",    "low",    "🟢"

# ─────────────────────────────────────────────  SIDEBAR
with st.sidebar:
    st.markdown("""
    <div style="padding:1rem 0 0.5rem;">
      <div class="sb-logo">⚙ Gear<span>Vision</span></div>
      <div style="font-family:'Share Tech Mono',monospace;font-size:0.56rem;color:#2a3a4a;
                  letter-spacing:0.28em;text-align:center;margin-top:3px;">
        DEFECT INSPECTION SYSTEM
      </div>
    </div>
    <div class="sb-divider"></div>
    <div style="font-family:'Share Tech Mono',monospace;font-size:0.58rem;color:#4a5a6a;
                letter-spacing:0.2em;text-transform:uppercase;margin-bottom:0.5rem;padding-left:2px;">
      Detection Parameters
    </div>
    """, unsafe_allow_html=True)

    conf_thresh = st.slider("Confidence Threshold", 0.10, 0.90, 0.20, 0.05)
    iou_thresh  = st.slider("IoU (NMS) Threshold",  0.10, 0.70, 0.30, 0.05)
    img_size    = st.select_slider("Image Resolution", options=[640, 960, 1280], value=1280)
    use_tta     = st.toggle("Test-Time Augmentation", value=True)

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'Share Tech Mono',monospace;font-size:0.58rem;color:#4a5a6a;
                letter-spacing:0.2em;text-transform:uppercase;margin-bottom:0.5rem;padding-left:2px;">
      Model Specifications
    </div>
    """, unsafe_allow_html=True)
    for k, v in [("Architecture","YOLOv8m-OBB"),("Classes","3"),("mAP50","99.4%"),
                 ("Precision","98.7%"),("Recall","99.7%"),("Parameters","26.4M")]:
        st.markdown(f'<div class="sb-stat"><span class="sb-stat-key">{k}</span>'
                    f'<span class="sb-stat-val">{v}</span></div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'Share Tech Mono',monospace;font-size:0.56rem;color:#2a3a4a;
                text-align:center;line-height:2;">
      🤗 HuggingFace<br>Shubhamv12/Gear_Detection
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────  HERO
st.markdown("""
<div class="hero-wrap">
  <div class="hero-bg-lines"></div>
  <span class="hero-tag">⚙ Mechanical Engineering · AI Quality Control System</span>
  <div class="hero-title">
    <span class="gear">GEAR</span><span class="vision">VISION</span> <span class="ai">AI</span>
  </div>
  <div class="hero-sub">Oriented Bounding Box &nbsp;·&nbsp; Real-Time Defect Detection &nbsp;·&nbsp; Industrial QC</div>
  <div class="hero-badge-row">
    <span class="hero-badge">OBB Detection</span>
    <span class="hero-badge">mAP50 99.4%</span>
    <span class="hero-badge">YOLOv8m</span>
    <span class="hero-badge">3 Defect Classes</span>
  </div>
</div>
<div class="sep-line"></div>
""", unsafe_allow_html=True)

st.markdown('<div style="height:1rem;"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────  LOAD MODEL
with st.spinner("Initializing detection engine…"):
    try:
        model = load_model()
        st.markdown("""
        <div style="background:#0a1a0e;border:1px solid #1a4a20;border-left:3px solid #00e676;
                    padding:0.55rem 1rem;margin-bottom:0.5rem;
                    clip-path:polygon(0 0,calc(100% - 10px) 0,100% 10px,100% 100%,0 100%);">
          <span style="font-family:'Share Tech Mono',monospace;font-size:0.7rem;color:#00e676;letter-spacing:0.1em;">
            ◆ ENGINE ONLINE — YOLOv8m-OBB ready for inspection
          </span>
        </div>""", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Engine failure: {e}")
        st.stop()

st.markdown('<div style="height:0.5rem;"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────  UPLOAD
col_up, col_info = st.columns([2, 1], gap="large")

with col_up:
    st.markdown("""
    <div style="font-family:'Rajdhani',sans-serif;font-weight:600;font-size:1.1rem;
                letter-spacing:0.1em;color:#8ab4d4;text-transform:uppercase;margin-bottom:0.5rem;">
      ◆ Upload Gear Image
    </div>""", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload", type=["jpg","jpeg","png","bmp","webp"],
                                label_visibility="collapsed")

with col_info:
    st.markdown("""
    <div style="font-family:'Rajdhani',sans-serif;font-weight:600;font-size:1.1rem;
                letter-spacing:0.1em;color:#8ab4d4;text-transform:uppercase;margin-bottom:0.5rem;">
      ◆ Detectable Defects
    </div>
    <div class="card card-steel">
      <div class="defect-info-row">
        <div class="defect-dot" style="background:#ff4444;"></div>
        <div><div class="defect-info-name">Break</div>
             <div class="defect-info-desc">Broken or chipped gear tooth</div></div>
      </div>
      <div class="defect-info-row">
        <div class="defect-dot" style="background:#00e676;"></div>
        <div><div class="defect-info-name">Lack</div>
             <div class="defect-info-desc">Missing material or tooth</div></div>
      </div>
      <div class="defect-info-row" style="border-bottom:none;">
        <div class="defect-dot" style="background:#00d4ff;"></div>
        <div><div class="defect-info-name">Scratch</div>
             <div class="defect-info-desc">Surface scratch or wear mark</div></div>
      </div>
      <div style="margin-top:0.75rem;font-family:'Share Tech Mono',monospace;
                  font-size:0.58rem;color:#4a5a6a;line-height:1.9;">
        JPG · PNG · BMP · WebP<br>
        Best: top-view, uniform lighting<br>
        Handles any rotation angle
      </div>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────  INFERENCE
if uploaded is not None:
    st.markdown('<div class="sep-line" style="margin:1rem 0;"></div>', unsafe_allow_html=True)

    suffix = Path(uploaded.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    img_pil = Image.open(tmp_path).convert("RGB")
    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    h, w    = img_bgr.shape[:2]

    scan_ph = st.empty()
    scan_ph.markdown("""
    <div style="text-align:center;padding:1.5rem;">
      <div class="scan-text">⚙ SCANNING GEAR SURFACE · PLEASE WAIT...</div>
    </div>""", unsafe_allow_html=True)
    time.sleep(0.4)

    t0      = time.time()
    result  = run_inference(model, tmp_path, conf_thresh, iou_thresh, img_size, use_tta)
    elapsed = time.time() - t0

    drawn_bgr, detections = draw_detections(result, img_bgr)
    drawn_rgb = cv2.cvtColor(drawn_bgr, cv2.COLOR_BGR2RGB)
    scan_ph.empty()

    total        = len(detections)
    sev_label, sev_cls, sev_icon = get_severity(total, detections)
    class_counts = Counter(d["class"] for d in detections)
    avg_conf     = np.mean([d["conf"] for d in detections]) if detections else 0.0

    # Metrics
    for col, val, lbl in zip(
        st.columns(4),
        [str(total), f"{elapsed*1000:.0f}ms", f"{w}×{h}", f"{avg_conf:.0%}" if detections else "—"],
        ["Defects Found", "Inference Time", "Resolution", "Avg Confidence"]
    ):
        with col:
            st.markdown(f'<div class="metric-box"><div class="metric-val">{val}</div>'
                        f'<div class="metric-lbl">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown('<div style="height:0.75rem;"></div>', unsafe_allow_html=True)

    # Images
    sev_card = {"high":"card-danger","medium":"card-warn","low":"card-ok","normal":"card-ok"}
    img_col, res_col = st.columns(2, gap="large")
    with img_col:
        st.markdown('<div class="card"><div class="img-label">Original Image</div>', unsafe_allow_html=True)
        st.image(img_pil, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with res_col:
        st.markdown(f'<div class="card {sev_card.get(sev_cls,"")}">'
                    f'<div class="img-label">Detection Result</div>', unsafe_allow_html=True)
        st.image(drawn_rgb, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:0.75rem;"></div>', unsafe_allow_html=True)

    # Breakdown + Verdict
    det_col, ver_col = st.columns([3, 2], gap="large")
    cls_color = {"break":"#ff4444","lack":"#00e676","scratch":"#00d4ff"}

    with det_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="sec-hdr">Defect Breakdown</div>', unsafe_allow_html=True)
        if not detections:
            st.markdown('<div class="no-defect-msg">◆ PASS — Gear surface is NORMAL</div>',
                        unsafe_allow_html=True)
        else:
            for cls_name, count in class_counts.items():
                confs = [d["conf"] for d in detections if d["class"] == cls_name]
                avg   = np.mean(confs)
                cc    = cls_color.get(cls_name, "#d4a843")
                st.markdown(
                    f'<div class="defect-row">'
                    f'<span class="badge b-{cls_name}">◆ {cls_name.upper()}</span>'
                    f'<span class="defect-count">{count} instance{"s" if count>1 else ""}</span>'
                    f'<span class="defect-avg">{avg:.0%}</span></div>'
                    f'<div class="cbar-wrap"><div class="cbar" style="width:{avg*100:.0f}%;background:{cc};"></div></div>',
                    unsafe_allow_html=True,
                )
        st.markdown('</div>', unsafe_allow_html=True)

        if detections:
            st.markdown('<div class="card" style="margin-top:0.5rem;">', unsafe_allow_html=True)
            st.markdown('<div class="sec-hdr">All Detections</div>', unsafe_allow_html=True)
            txt_color = {"break":"#ff7070","lack":"#50ffaa","scratch":"#60ddff"}
            for i, det in enumerate(detections):
                st.markdown(
                    f'<div class="det-row">'
                    f'<span class="det-idx">#{i+1}</span>'
                    f'<span class="badge b-{det["class"]}" style="font-size:0.6rem;padding:0.12rem 0.5rem;">◆ {det["class"]}</span>'
                    f'<span class="det-conf" style="color:{txt_color.get(det["class"],"#d4a843")}">{det["conf"]:.0%}</span>'
                    f'</div>', unsafe_allow_html=True,
                )
            st.markdown('</div>', unsafe_allow_html=True)

    with ver_col:
        sev_colors = {"high":"#ff4444","medium":"#ffd740","low":"#00e676","normal":"#00e676"}
        action_map = {
            "high"  : "REJECT — Discard immediately",
            "medium": "INSPECT — Manual review required",
            "low"   : "MONITOR — Flag for next cycle",
            "normal": "PASS — Clear for production",
        }
        sc = sev_colors[sev_cls]
        st.markdown(
            f'<div class="verdict-box">'
            f'<div class="sec-hdr" style="justify-content:center;">Quality Verdict</div>'
            f'<div style="font-size:3rem;line-height:1;margin-bottom:0.5rem;">{sev_icon}</div>'
            f'<div class="verdict-label" style="color:{sc};">{sev_label}</div>'
            f'<div class="verdict-bar" style="background:{sc};"></div>'
            f'<div class="verdict-action">{action_map[sev_cls]}</div>'
            f'</div>', unsafe_allow_html=True,
        )

    os.unlink(tmp_path)

else:
    st.markdown("""
    <div class="empty-state">
      <div class="empty-icon">⚙</div>
      <div class="empty-title">Awaiting Input</div>
      <div class="empty-sub">Upload a gear image above to begin defect analysis</div>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────  FOOTER
st.markdown('<div class="sep-line" style="margin-top:2rem;"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="footer-outer">
  <div class="footer-label">◆ &nbsp; Developed By &nbsp; ◆</div>
  <div class="dev-row">

    <div class="dev-card">
      <div class="dev-avatar av-gold">SV</div>
      <div>
        <div class="dev-name">Shubham Verma</div>
        <div class="dev-role">MECHANICAL ENGINEER</div>
      </div>
    </div>

    <div class="dev-card">
      <div class="dev-avatar av-steel">SP</div>
      <div>
        <div class="dev-name">Shubham Pal</div>
        <div class="dev-role">MECHANICAL ENGINEER</div>
      </div>
    </div>

  </div>
  <div class="footer-copy">
    GearVision AI &nbsp;·&nbsp; YOLOv8m-OBB &nbsp;·&nbsp; mAP50 99.4%
    &nbsp;·&nbsp; Shubhamv12/Gear_Detection &nbsp;·&nbsp; Built with Streamlit
  </div>
</div>
""", unsafe_allow_html=True)
