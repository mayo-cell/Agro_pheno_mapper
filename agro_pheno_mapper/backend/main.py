"""
FastAPI Backend Server for AgroPheno Mapper.
Connects the Gonzalez & Woods DIP engine, SQLite database layer, and separate HTML/CSS/JS frontend.
"""

import csv
import io
import os
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .database import Database
from .dip_engine import generate_sample_drone_plot, process_drone_image

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
DATA_DIR = Path(os.getenv("AGROPHENO_DATA_DIR", str(BASE_DIR / "data")))
OUTPUTS_DIR = DATA_DIR / "outputs"
UPLOADS_DIR = DATA_DIR / "uploads"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="AgroPheno Mapper API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
)

db = Database(DATA_DIR / "agropheno.db")
MAX_UPLOAD_SIZE = 10 * 1024 * 1024


def _remove_file(path: Optional[str | Path]) -> None:
    if not path:
        return
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass


def _cleanup_scan_files(scan: dict) -> None:
    for key in ("raw_path", "enhanced_path", "heatmap_path", "mask_path"):
        _remove_file(scan.get(key))

    raw_path = scan.get("raw_path")
    if raw_path:
        uid = Path(raw_path).stem.rsplit("_", 1)[0]
        for upload in UPLOADS_DIR.glob(f"{uid}_*"):
            _remove_file(upload)

app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")
if FRONTEND_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR)), name="assets")


class PlotCreate(BaseModel):
    name: str
    crop_type: str = "Row Crop"
    location: str = ""


def _serialize_scan(scan: dict) -> dict:
    """Attach clean relative URLs for frontend image rendering."""
    return {
        **scan,
        "raw_url": f"/outputs/{Path(scan['raw_path']).name}",
        "enhanced_url": f"/outputs/{Path(scan['enhanced_path']).name}",
        "heatmap_url": f"/outputs/{Path(scan['heatmap_path']).name}",
        "mask_url": f"/outputs/{Path(scan['mask_path']).name}",
    }


@app.get("/")
def serve_frontend():
    index_file = FRONTEND_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend index.html not found")
    return FileResponse(str(index_file))


@app.get("/api/health")
def health_check():
    return {"status": "ok", "service": "AgroPheno Mapper"}


@app.get("/api/plots")
def list_plots():
    return {"plots": db.list_plots()}


@app.post("/api/plots")
def create_plot(payload: PlotCreate):
    try:
        plot = db.create_plot(payload.name, payload.crop_type, payload.location)
        return {"plot": plot}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not create plot: {exc}")


@app.post("/api/analyze")
async def analyze_upload(
    file: UploadFile = File(...),
    plot_name: str = Form("Plot A-1 (Demo)"),
    crop_type: str = Form("Row Crop"),
    gamma: float = Form(1.0),
    sharpen: bool = Form(True),
):
    uid = uuid.uuid4().hex[:10]
    ext = Path(file.filename or "upload.png").suffix.lower() or ".png"
    if ext not in {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}:
        raise HTTPException(status_code=400, detail="Unsupported image format")
    upload_path = UPLOADS_DIR / f"{uid}_input{ext}"

    content = await file.read(MAX_UPLOAD_SIZE + 1)
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="Uploaded file exceeds the 10 MB limit")
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    upload_path.write_bytes(content)

    raw_out = OUTPUTS_DIR / f"{uid}_raw.png"
    enhanced_out = OUTPUTS_DIR / f"{uid}_enhanced.png"
    heatmap_out = OUTPUTS_DIR / f"{uid}_heatmap.png"
    mask_out = OUTPUTS_DIR / f"{uid}_mask.png"

    try:
        metrics = process_drone_image(
            str(upload_path),
            str(raw_out),
            str(enhanced_out),
            str(heatmap_out),
            str(mask_out),
            gamma=gamma,
            sharpen=sharpen,
        )
    except ValueError as exc:
        _remove_file(upload_path)
        raise HTTPException(status_code=400, detail=str(exc))
    finally:
        if upload_path.exists():
            _remove_file(upload_path)

    plot_id = db.get_or_create_plot(plot_name, crop_type)
    scan = db.insert_scan(
        {
            "plot_id": plot_id,
            "filename": file.filename or f"upload{ext}",
            "gamma": gamma,
            "otsu_threshold": metrics["otsu_threshold"],
            "canopy_coverage_pct": metrics["canopy_coverage_pct"],
            "stressed_area_pct": metrics["stressed_area_pct"],
            "bare_soil_pct": metrics["bare_soil_pct"],
            "mean_exg": metrics["mean_exg"],
            "mean_vari": metrics["mean_vari"],
            "cluster_count": metrics["cluster_count"],
            "avg_compactness": metrics["avg_compactness"],
            "row_uniformity_pct": metrics["row_uniformity_pct"],
            "raw_path": str(raw_out),
            "enhanced_path": str(enhanced_out),
            "heatmap_path": str(heatmap_out),
            "mask_path": str(mask_out),
        }
    )
    return {"scan": _serialize_scan(scan)}


@app.post("/api/analyze-sample")
def analyze_sample(
    plot_name: str = Form("Plot A-1 (Demo)"),
    gamma: float = Form(1.0),
    sharpen: bool = Form(True),
):
    uid = uuid.uuid4().hex[:10]
    sample_path = UPLOADS_DIR / f"{uid}_sample_drone.png"

    raw_out = OUTPUTS_DIR / f"{uid}_raw.png"
    enhanced_out = OUTPUTS_DIR / f"{uid}_enhanced.png"
    heatmap_out = OUTPUTS_DIR / f"{uid}_heatmap.png"
    mask_out = OUTPUTS_DIR / f"{uid}_mask.png"

    try:
        generate_sample_drone_plot(str(sample_path))
        metrics = process_drone_image(
            str(sample_path),
            str(raw_out),
            str(enhanced_out),
            str(heatmap_out),
            str(mask_out),
            gamma=gamma,
            sharpen=sharpen,
        )
    finally:
        _remove_file(sample_path)

    plot_id = db.get_or_create_plot(plot_name, "Maize / Row Crop")
    scan = db.insert_scan(
        {
            "plot_id": plot_id,
            "filename": "sample_drone_plot.png",
            "gamma": gamma,
            "otsu_threshold": metrics["otsu_threshold"],
            "canopy_coverage_pct": metrics["canopy_coverage_pct"],
            "stressed_area_pct": metrics["stressed_area_pct"],
            "bare_soil_pct": metrics["bare_soil_pct"],
            "mean_exg": metrics["mean_exg"],
            "mean_vari": metrics["mean_vari"],
            "cluster_count": metrics["cluster_count"],
            "avg_compactness": metrics["avg_compactness"],
            "row_uniformity_pct": metrics["row_uniformity_pct"],
            "raw_path": str(raw_out),
            "enhanced_path": str(enhanced_out),
            "heatmap_path": str(heatmap_out),
            "mask_path": str(mask_out),
        }
    )
    return {"scan": _serialize_scan(scan)}


@app.get("/api/scans")
def list_scans(plot_id: Optional[int] = None, limit: int = 50):
    scans = [_serialize_scan(s) for s in db.list_scans(plot_id=plot_id, limit=limit)]
    return {"scans": scans}


@app.get("/api/scans/{scan_id}")
def get_scan(scan_id: int):
    scan = db.get_scan(scan_id)
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")
    return {"scan": _serialize_scan(scan)}


@app.delete("/api/scans/{scan_id}")
def delete_scan(scan_id: int):
    scan = db.get_scan(scan_id)
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")
    deleted = db.delete_scan(scan_id)
    if deleted:
        _cleanup_scan_files(scan)
    return {"deleted": True, "scan_id": scan_id}


@app.get("/api/scans/{scan_id}/export-csv")
def export_scan_csv(scan_id: int):
    scan = db.get_scan(scan_id)
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "Scan ID",
            "Plot Name",
            "Filename",
            "Gamma",
            "Otsu Threshold (ExG)",
            "Canopy Coverage (%)",
            "Stressed Area (%)",
            "Bare Soil (%)",
            "Mean ExG",
            "Mean VARI",
            "Cluster Count",
            "Avg Compactness",
            "Row Uniformity (%)",
            "Timestamp",
        ]
    )
    writer.writerow(
        [
            scan["id"],
            scan["plot_name"],
            scan["filename"],
            scan["gamma"],
            scan["otsu_threshold"],
            scan["canopy_coverage_pct"],
            scan["stressed_area_pct"],
            scan["bare_soil_pct"],
            scan["mean_exg"],
            scan["mean_vari"],
            scan["cluster_count"],
            scan["avg_compactness"],
            scan["row_uniformity_pct"],
            scan["created_at"],
        ]
    )
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=scan_{scan_id}_metrics.csv"},
    )
