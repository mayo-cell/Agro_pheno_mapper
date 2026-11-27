# AgroPheno Mapper — Micro-Plot Drone Crop Stress & Canopy Analyzer

Full-stack Digital Image Processing (DIP) application built with a **separated architecture**:
- **Backend (`backend/main.py` + `backend/dip_engine.py`):** Python + FastAPI + OpenCV/NumPy/Matplotlib
- **Database (`backend/database.py`):** SQLite (`plots` and `scans` relational schema)
- **Frontend (`frontend/`):** Vanilla HTML5, CSS3, and JavaScript (`index.html`, `styles.css`, `app.js`)

## Syllabus Alignment (Gonzalez & Woods, 4th Edition)
1. **Unit 2 — Spatial Domain Enhancement:** Power-law (Gamma) transformation ($s = c \cdot r^\gamma$), $3\times 3$ median smoothing, and unsharp masking.
2. **Unit 3 — Color Image Processing:** Normalized RGB chromaticity, Excess Green Index ($ExG = 2g - r - b$), VARI, and exact HSI Hue conversion for chlorosis (yellowing) detection.
3. **Unit 4 — Image Segmentation:** Otsu's optimal global thresholding on ExG and contour/edge boundary extraction.
4. **Unit 5 — Representation & Description:** Morphological opening/closing, connected-component clustering, compactness ($4\pi A / P^2$), and row uniformity metrics.

## Quick Start
```bash
cd agro_pheno_mapper
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000
```
Open `http://localhost:8000` in your browser and click **Load Sample Drone Plot** or upload your own top-down crop photo.

## Running Tests
```bash
python3 -m unittest discover -s tests -v
```
