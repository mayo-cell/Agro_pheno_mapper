# AgroPheno Mapper — Micro-Plot Drone Crop Stress & Canopy Analyzer

Full-stack Digital Image Processing (DIP) application built with a **separated architecture**:
- **Backend (`backend/main.py` + `backend/dip_engine.py`):** Python + FastAPI + OpenCV/NumPy/Matplotlib
- **Database (`backend/database.py`):** SQLite (`plots` and `scans` relational schema)
- **Frontend (`frontend/`):** Vanilla HTML5, CSS3, and JavaScript (`index.html`, `styles.css`, `app.js`)


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
