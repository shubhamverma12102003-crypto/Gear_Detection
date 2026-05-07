#old code
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

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="GearVision AI",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS  — Industrial Dark Theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;900&family=DM+Mono:wght@300;400;500&family=Inter:wght@300;400;500&display=swap');

/* ── Base ─────────────────────────────────── */
:root {
    --bg:        #0a0c0f;
    --surface:   #111418;
    --panel:     #181c22;
    --border:    #2a2f38;
    --accent:    #00e5ff;
    --accent2:   #ff6b35;
    --accent3:   #39ff14;
    --text:      #e8eaf0;
    --muted:     #6b7280;
    --danger:    #ff3b30;
    --warning:   #ffd60a;
    --ok:        #30d158;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text);
    font-family: 'Inter', sans-serif;
}

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}

/* hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

/* ── Typography ───────────────────────────── */
.title-hero {
    font-family: 'Orbitron', monospace;
    font-weight: 900;
    font-size: clamp(2rem, 5vw, 3.5rem);
    background: linear-gradient(135deg, var(--accent) 0%, #0080ff 50%, var(--accent2) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: 0.08em;
    line-height: 1.1;
    text-align: center;
    animation: shimmer 4s ease-in-out infinite alternate;
}

@keyframes shimmer {
    0%  { filter: brightness(1); }
    100%{ filter: brightness(1.3) drop-shadow(0 0 20px var(--accent)); }
}

.subtitle {
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    color: var(--muted);
    letter-spacing: 0.25em;
    text-align: center;
    text-transform: uppercase;
}

/* ── Cards ────────────────────────────────── */
.card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin: 0.75rem 0;
    position: relative;
    overflow: hidden;
}

.card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent), transparent);
}

.card-danger::before  { background: linear-gradient(90deg, var(--danger), transparent); }
.card-warning::before { background: linear-gradient(90deg, var(--warning), transparent); }
.card-ok::before      { background: linear-gradient(90deg, var(--ok), transparent); }

/* ── Metric blocks ────────────────────────── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.75rem;
    margin: 1rem 0;
}

.metric-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}

.metric-box .val {
    font-family: 'Orbitron', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--accent);
}

.metric-box .lbl {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-top: 0.25rem;
}

/* ── Detection badges ─────────────────────── */
.badge {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-weight: 500;
    letter-spacing: 0.05em;
}

.badge-break   { background: rgba(255,59,48,0.15);  border: 1px solid var(--danger);  color: #ff6b6b; }
.badge-lack    { background: rgba(48,209,88,0.15);  border: 1px solid var(--ok);     color: #5effa0; }
.badge-scratch { background: rgba(0,229,255,0.15);  border: 1px solid var(--accent); color: var(--accent); }

/* ── Severity bar ─────────────────────────── */
.severity-label {
    font-family: 'Orbitron', monospace;
    font-size: 1.1rem;
    font-weight: 600;
    letter-spacing: 0.1em;
}

.severity-high    { color: var(--danger);  }
.severity-medium  { color: var(--warning); }
.severity-low     { color: var(--ok);      }
.severity-normal  { color: var(--accent3); }

/* ── Progress / confidence bars ───────────── */
.conf-bar-wrap {
    background: var(--surface);
    border-radius: 4px;
    height: 6px;
    overflow: hidden;
    margin-top: 4px;
}
.conf-bar {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, var(--accent), #0080ff);
}

/* ── Upload zone ──────────────────────────── */
[data-testid="stFileUploader"] {
    border: 2px dashed var(--border) !important;
    border-radius: 12px !important;
    background: var(--panel) !important;
    transition: border-color 0.3s;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent) !important;
}

/* ── Buttons ──────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent), #0080ff) !important;
    color: #000 !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 0.75rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.15em !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.3s !important;
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(0,229,255,0.35) !important;
}

/* ── Sidebar items ────────────────────────── */
.sidebar-stat {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin: 0.4rem 0;
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: var(--muted);
}
.sidebar-stat span {
    color: var(--accent);
    float: right;
    font-weight: 500;
}

/* ── Divider ──────────────────────────────── */
.glow-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--accent), transparent);
    margin: 1.5rem 0;
    opacity: 0.4;
}

/* ── Scan animation overlay text ─────────── */
.scanning-text {
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    color: var(--accent);
    letter-spacing: 0.2em;
    text-align: center;
    animation: blink 1s step-end infinite;
}
@keyframes blink { 50% { opacity: 0; } }

/* ── Streamlit overrides ──────────────────── */
.stSlider [data-baseweb="slider"] { background: var(--border) !important; }
.stSelectbox div[data-baseweb="select"] > div { background: var(--panel) !important; border-color: var(--border) !important; }
.stRadio label { font-family: 'DM Mono', monospace; font-size: 0.8rem; color: var(--muted); }
p, li, label { color: var(--text) !important; }
h1, h2, h3 { font-family: 'Orbitron', monospace !important; color: var(--text) !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
MODEL_URL  = "https://huggingface.co/ritesht04/Gear_Detection/resolve/main/best%20(3).pt"
MODEL_PATH = Path("best.pt")

CLASS_NAMES  = {0: "break", 1: "lack", 2: "scratch"}
CLASS_COLORS = {
    "break"  : (255, 59,  48),
    "lack"   : (48,  209, 88),
    "scratch": (0,   229, 255),
}
CLASS_EMOJI  = {"break": "🔴", "lack": "🟢", "scratch": "🔵"}

# ─────────────────────────────────────────────
#  MODEL LOADER
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_PATH.exists():
        with requests.get(MODEL_URL, stream=True) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            downloaded = 0
            bar = st.progress(0, text="Downloading model from HuggingFace...")
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        bar.progress(downloaded / total,
                                     text=f"Downloading model… {downloaded/1e6:.1f} / {total/1e6:.1f} MB")
            bar.empty()
    return YOLO(str(MODEL_PATH))


# ─────────────────────────────────────────────
#  INFERENCE
# ─────────────────────────────────────────────
def run_inference(model, img_path, conf, iou, imgsz, augment):
    results = model.predict(
        source=str(img_path),
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device="cpu",
        augment=augment,
        verbose=False,
    )
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

            pts = result.obb.xyxyxyxy[i].cpu().numpy()
            pts = pts.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(drawn, [pts], isClosed=True, color=color, thickness=3)

            x1 = int(result.obb.xyxyxyxy[i].cpu().numpy()[:, 0].min())
            y1 = int(result.obb.xyxyxyxy[i].cpu().numpy()[:, 1].min())

            label     = f"{cls_name} {conf_val:.0%}"
            font      = cv2.FONT_HERSHEY_SIMPLEX
            fs        = 0.6
            th        = 2
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
        return "HIGH", "high", "🔴"
    elif total >= 2:
        return "MEDIUM", "medium", "🟡"
    else:
        return "LOW", "low", "🟢"


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="title-hero" style="font-size:1.4rem;">⚙️ GearVision</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle" style="font-size:0.65rem;">Industrial Defect AI</p>', unsafe_allow_html=True)
    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

    st.markdown("#### Detection Settings")

    conf_thresh = st.slider("Confidence Threshold", 0.10, 0.90, 0.20, 0.05,
                            help="Minimum confidence to show a detection")
    iou_thresh  = st.slider("IoU (NMS) Threshold", 0.10, 0.70, 0.30, 0.05,
                            help="Overlap threshold for box merging")
    img_size    = st.select_slider("Image Resolution", options=[640, 960, 1280], value=1280)
    use_tta     = st.toggle("Test-Time Augmentation", value=True,
                            help="Improves accuracy, slightly slower")

    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

    st.markdown("#### Model Info")
    for label, val in [
        ("Architecture", "YOLOv8m-OBB"),
        ("Classes", "3"),
        ("mAP50", "99.4 %"),
        ("Precision", "98.7 %"),
        ("Recall", "99.7 %"),
        ("Params", "26.4 M"),
    ]:
        st.markdown(
            f'<div class="sidebar-stat">{label}<span>{val}</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-family:DM Mono,monospace;font-size:0.65rem;color:#4b5563;text-align:center;">'
        'Model hosted on 🤗 HuggingFace<br>ritesht04/Gear_Detection</p>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
#  MAIN HEADER
# ─────────────────────────────────────────────
st.markdown('<div style="height:1rem;"></div>', unsafe_allow_html=True)
st.markdown('<h1 class="title-hero">GEARVISION AI</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Oriented Bounding Box · Real-Time Defect Detection · Industrial QC</p>',
    unsafe_allow_html=True,
)
st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────────
with st.spinner("Initializing GearVision AI engine…"):
    try:
        model = load_model()
        st.markdown(
            '<div class="card card-ok" style="padding:0.6rem 1rem;">'
            '<p style="font-family:DM Mono,monospace;font-size:0.75rem;margin:0;color:#30d158;">'
            '⚡ Model ready — YOLOv8m-OBB loaded from HuggingFace</p></div>',
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

st.markdown('<div style="height:0.5rem;"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  FILE UPLOAD
# ─────────────────────────────────────────────
col_up, col_info = st.columns([2, 1], gap="large")

with col_up:
    st.markdown("### Upload Gear Image")
    uploaded = st.file_uploader(
        "Drag & drop or click to upload",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        label_visibility="collapsed",
    )

with col_info:
    st.markdown("### Detectable Defects")
    st.markdown(
        '<div class="card">'
        '<p style="font-family:DM Mono,monospace;font-size:0.8rem;margin:0.3rem 0;">'
        '<span class="badge badge-break">BREAK</span>'
        ' &nbsp;Broken or chipped tooth</p>'
        '<p style="font-family:DM Mono,monospace;font-size:0.8rem;margin:0.3rem 0;">'
        '<span class="badge badge-lack">LACK</span>'
        ' &nbsp;Missing material / tooth</p>'
        '<p style="font-family:DM Mono,monospace;font-size:0.8rem;margin:0.3rem 0;">'
        '<span class="badge badge-scratch">SCRATCH</span>'
        ' &nbsp;Surface scratch / mark</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="card" style="padding:1rem;">'
        '<p style="font-family:DM Mono,monospace;font-size:0.7rem;color:#6b7280;margin:0;">'
        'Supports JPG · PNG · BMP · WebP<br>'
        'Best results: top-view, clear lighting<br>'
        'OBB model handles any rotation angle</p>'
        '</div>',
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
#  INFERENCE SECTION
# ─────────────────────────────────────────────
if uploaded is not None:
    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

    # Save to temp file
    suffix = Path(uploaded.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    img_pil = Image.open(tmp_path).convert("RGB")
    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    h, w    = img_bgr.shape[:2]

    # ── Scanning animation
    scan_ph = st.empty()
    scan_ph.markdown(
        '<div style="text-align:center;padding:1rem;">'
        '<p class="scanning-text">⚙ SCANNING GEAR SURFACE...</p></div>',
        unsafe_allow_html=True,
    )
    time.sleep(0.4)

    # ── Run inference
    t0     = time.time()
    result = run_inference(model, tmp_path, conf_thresh, iou_thresh, img_size, use_tta)
    elapsed = time.time() - t0

    drawn_bgr, detections = draw_detections(result, img_bgr)
    drawn_rgb = cv2.cvtColor(drawn_bgr, cv2.COLOR_BGR2RGB)
    scan_ph.empty()

    # ─────────────────────────────────────────
    #  RESULTS LAYOUT
    # ─────────────────────────────────────────
    total       = len(detections)
    sev_label, sev_cls, sev_icon = get_severity(total, detections)
    class_counts = Counter(d["class"] for d in detections)

    # Top metrics row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            f'<div class="metric-box"><div class="val">{total}</div>'
            f'<div class="lbl">Defects Found</div></div>',
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f'<div class="metric-box"><div class="val">{elapsed*1000:.0f}ms</div>'
            f'<div class="lbl">Inference Time</div></div>',
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f'<div class="metric-box"><div class="val">{w}×{h}</div>'
            f'<div class="lbl">Image Resolution</div></div>',
            unsafe_allow_html=True,
        )
    with m4:
        avg_conf = np.mean([d["conf"] for d in detections]) if detections else 0.0
        st.markdown(
            f'<div class="metric-box"><div class="val">{avg_conf:.0%}</div>'
            f'<div class="lbl">Avg Confidence</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div style="height:0.5rem;"></div>', unsafe_allow_html=True)

    # Image comparison columns
    img_col, res_col = st.columns(2, gap="large")

    with img_col:
        st.markdown(
            '<div class="card"><p style="font-family:DM Mono,monospace;font-size:0.7rem;'
            'color:#6b7280;margin:0 0 0.5rem 0;letter-spacing:0.1em;">ORIGINAL IMAGE</p>',
            unsafe_allow_html=True,
        )
        st.image(img_pil, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with res_col:
        st.markdown(
            '<div class="card card-' + ("danger" if sev_cls == "high" else sev_cls if sev_cls != "normal" else "ok") + '">'
            '<p style="font-family:DM Mono,monospace;font-size:0.7rem;'
            'color:#6b7280;margin:0 0 0.5rem 0;letter-spacing:0.1em;">DETECTION RESULT</p>',
            unsafe_allow_html=True,
        )
        st.image(drawn_rgb, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div style="height:0.5rem;"></div>', unsafe_allow_html=True)

    # Severity + breakdown
    detail_col, sev_col = st.columns([3, 2], gap="large")

    with detail_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            '<p style="font-family:DM Mono,monospace;font-size:0.7rem;'
            'color:#6b7280;margin:0 0 0.75rem 0;letter-spacing:0.1em;">DEFECT BREAKDOWN</p>',
            unsafe_allow_html=True,
        )

        if not detections:
            st.markdown(
                '<p style="font-family:Orbitron,monospace;font-size:1rem;color:#39ff14;">'
                '✅ No defects detected — Gear surface is NORMAL</p>',
                unsafe_allow_html=True,
            )
        else:
            for cls_name, count in class_counts.items():
                confs    = [d["conf"] for d in detections if d["class"] == cls_name]
                avg      = np.mean(confs)
                badge_cls = f"badge-{cls_name}"
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:1rem;margin:0.6rem 0;">'
                    f'<span class="badge {badge_cls}">{CLASS_EMOJI[cls_name]} {cls_name.upper()}</span>'
                    f'<span style="font-family:DM Mono,monospace;font-size:0.8rem;color:#9ca3af;">'
                    f'{count} instance{"s" if count>1 else ""}</span>'
                    f'<span style="font-family:Orbitron,monospace;font-size:0.85rem;color:#e8eaf0;margin-left:auto;">'
                    f'{avg:.0%}</span></div>'
                    f'<div class="conf-bar-wrap"><div class="conf-bar" style="width:{avg*100:.0f}%;'
                    f'background:linear-gradient(90deg,{"#ff3b30" if cls_name=="break" else "#30d158" if cls_name=="lack" else "#00e5ff"},transparent);"></div></div>',
                    unsafe_allow_html=True,
                )
        st.markdown("</div>", unsafe_allow_html=True)

    with sev_col:
        sev_color_map = {
            "high"  : "#ff3b30",
            "medium": "#ffd60a",
            "low"   : "#30d158",
            "normal": "#39ff14",
        }
        sev_color = sev_color_map[sev_cls]
        action_map = {
            "high"  : "REJECT — Gear must be discarded",
            "medium": "INSPECT — Manual review required",
            "low"   : "MONITOR — Flag for next cycle",
            "normal": "PASS — Clear for production",
        }
        st.markdown(
            f'<div class="card" style="text-align:center;">'
            f'<p style="font-family:DM Mono,monospace;font-size:0.7rem;color:#6b7280;'
            f'letter-spacing:0.1em;margin:0 0 0.75rem 0;">QUALITY VERDICT</p>'
            f'<p style="font-family:Orbitron,monospace;font-size:2.5rem;font-weight:900;'
            f'color:{sev_color};margin:0;line-height:1;">{sev_icon}</p>'
            f'<p class="severity-label severity-{sev_cls}" style="font-size:1.4rem;margin:0.5rem 0;">'
            f'{sev_label}</p>'
            f'<div style="height:3px;background:{sev_color};border-radius:2px;margin:0.75rem 0;'
            f'opacity:0.4;"></div>'
            f'<p style="font-family:DM Mono,monospace;font-size:0.72rem;color:#9ca3af;margin:0;">'
            f'{action_map[sev_cls]}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # All detection list
        if detections:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(
                '<p style="font-family:DM Mono,monospace;font-size:0.65rem;color:#6b7280;'
                'letter-spacing:0.1em;margin:0 0 0.5rem 0;">ALL DETECTIONS</p>',
                unsafe_allow_html=True,
            )
            for i, det in enumerate(detections):
                cls  = det["class"]
                conf = det["conf"]
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;align-items:center;'
                    f'padding:0.25rem 0;border-bottom:1px solid #1f2937;">'
                    f'<span style="font-family:DM Mono,monospace;font-size:0.7rem;color:#6b7280;">#{i+1}</span>'
                    f'<span class="badge badge-{cls}" style="font-size:0.65rem;padding:0.15rem 0.5rem;">{cls}</span>'
                    f'<span style="font-family:Orbitron,monospace;font-size:0.75rem;'
                    f'color:{"#ff6b6b" if cls=="break" else "#5effa0" if cls=="lack" else "#00e5ff"};">'
                    f'{conf:.0%}</span></div>',
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

    # Clean up temp file
    os.unlink(tmp_path)

else:
    # Empty state
    st.markdown(
        '<div class="card" style="text-align:center;padding:3rem;">'
        '<p style="font-family:Orbitron,monospace;font-size:3rem;margin:0;">⚙️</p>'
        '<p style="font-family:Orbitron,monospace;font-size:1rem;color:#4b5563;'
        'letter-spacing:0.1em;margin:0.5rem 0;">AWAITING INPUT</p>'
        '<p style="font-family:DM Mono,monospace;font-size:0.75rem;color:#374151;">'
        'Upload a gear image above to begin defect analysis</p>'
        '</div>',
        unsafe_allow_html=True,
    )

# Footer
st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
st.markdown(
    '<p style="font-family:DM Mono,monospace;font-size:0.65rem;color:#374151;text-align:center;">'
    'GearVision AI · YOLOv8m-OBB · mAP50 99.4% · Built with Streamlit</p>',
    unsafe_allow_html=True,
)
