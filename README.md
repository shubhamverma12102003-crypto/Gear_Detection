# ⚙️ GearVision AI — Gear Defect Detection

> Real-time industrial gear defect detection using YOLOv8m-OBB.
> Detects **Break**, **Lack**, and **Scratch** defects with **99.4% mAP50** accuracy.

---

## 🚀 Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

---

## 📁 Project Structure

```
gear-defect-app/
├── app.py               ← Main Streamlit application
├── requirements.txt     ← Python dependencies
└── README.md            ← This file
```

> **Note:** The model (`best.pt`) is automatically downloaded from HuggingFace at runtime.
> You do NOT need to include it in the repository.

---

## 🤗 Model

| Property     | Value                                      |
|--------------|--------------------------------------------|
| Architecture | YOLOv8m-OBB (Oriented Bounding Box)        |
| Classes      | break, lack, scratch                       |
| mAP50        | 99.4%                                      |
| Precision    | 98.7%                                      |
| Recall       | 99.7%                                      |
| Model Size   | ~53 MB                                     |
| HuggingFace  | ritesht04/Gear_Detection                   |

---

## 🛠️ Local Setup

### Step 1 — Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/gear-defect-app.git
cd gear-defect-app
```

### Step 2 — Create Virtual Environment
```bash
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### Step 3 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Run App
```bash
streamlit run app.py
```

App will open at `http://localhost:8501`

---

## ☁️ Deploy on Streamlit Cloud (Free)

### Step 1 — Push to GitHub

```bash
# Initialize git in your project folder
git init
git add app.py requirements.txt README.md
git commit -m "Initial commit: GearVision AI"

# Create repo on GitHub first, then:
git remote add origin https://github.com/YOUR_USERNAME/gear-defect-app.git
git branch -M main
git push -u origin main
```

### Step 2 — Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click **"New app"**
4. Fill in the form:
   - **Repository:** `YOUR_USERNAME/gear-defect-app`
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. Click **"Deploy!"**
6. Wait 2–3 minutes for first build
7. Your app is live at `https://YOUR_USERNAME-gear-defect-app-app-XXXX.streamlit.app`

> **Note:** First load may take 1–2 minutes as the model downloads from HuggingFace (~53 MB).
> Subsequent loads are instant (model is cached by Streamlit).

---

## 📦 Dependencies Explained

| Package                    | Version   | Purpose                               |
|----------------------------|-----------|---------------------------------------|
| streamlit                  | 1.35.0    | Web app framework                     |
| ultralytics                | 8.4.46    | YOLOv8 model loading and inference    |
| opencv-python-headless     | 4.9.0.80  | Image processing & box drawing        |
| Pillow                     | 10.3.0    | Image reading and format conversion   |
| numpy                      | 1.26.4    | Array operations for coordinates      |
| requests                   | 2.31.0    | Downloading model from HuggingFace    |

> `opencv-python-headless` is used instead of `opencv-python` because Streamlit Cloud
> does not have a display server. The headless version works without GUI dependencies.

---

## ⚙️ App Features

- 🤗 **Auto model download** from HuggingFace on first run
- 📁 **Drag & drop** image upload (JPG, PNG, BMP, WebP)
- 🎛️ **Adjustable parameters** — confidence, IoU, resolution, TTA
- 🎨 **Color-coded detections** — Red (break), Green (lack), Blue (scratch)
- 📊 **Real-time metrics** — defect count, inference time, confidence scores
- 🏭 **Quality verdict** — PASS / LOW / MEDIUM / HIGH severity
- 🌑 **Industrial dark theme** — Orbitron font, neon accents

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| App crashes on startup | Check `requirements.txt` versions match exactly |
| Model download fails | Check internet connection; HuggingFace URL must be public |
| `cv2` import error | Use `opencv-python-headless` not `opencv-python` in requirements |
| Low detection confidence | Lower confidence slider to 0.15–0.20 |
| Too many false positives | Raise confidence slider to 0.40–0.50 |
| Streamlit Cloud build fails | Check Python version — use 3.10 or 3.11 |

---

## 📄 License

MIT License — Free to use for academic and personal projects.

---

*Built with ❤️ using YOLOv8 + Streamlit*
