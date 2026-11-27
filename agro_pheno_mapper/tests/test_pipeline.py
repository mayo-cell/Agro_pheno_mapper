import tempfile
import unittest
from pathlib import Path
import numpy as np
from fastapi.testclient import TestClient

import backend.main as main_module
from backend.database import Database
from backend.dip_engine import (
    apply_gamma_correction,
    compute_vegetation_indices,
    generate_sample_drone_plot,
    otsu_segment_canopy,
    process_drone_image,
)
from backend.main import app


class TestAgroPhenoPipeline(unittest.TestCase):
    def test_gamma_and_vegetation_indices(self):
        # Pure green patch vs pure soil-like brown patch
        rgb = np.zeros((40, 40, 3), dtype=np.uint8)
        rgb[:20, :, :] = [40, 180, 40]   # Green canopy
        rgb[20:, :, :] = [120, 85, 60]   # Brown soil

        gamma_img = apply_gamma_correction(rgb, gamma=0.8)
        self.assertEqual(gamma_img.shape, rgb.shape)
        self.assertGreater(float(gamma_img.mean()), float(rgb.mean()))

        exg, vari, hue = compute_vegetation_indices(rgb)
        self.assertGreater(float(exg[:20].mean()), 0.3)
        self.assertLess(float(exg[20:].mean()), 0.0)

        mask, thresh = otsu_segment_canopy(exg)
        self.assertGreater(float(mask[:20].mean()), 200.0)
        self.assertLess(float(mask[20:].mean()), 20.0)
        self.assertTrue(-0.5 <= thresh <= 0.8)

    def test_end_to_end_sample_and_database(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            sample_img = tmp / "sample.png"
            generate_sample_drone_plot(str(sample_img))
            self.assertTrue(sample_img.exists())

            metrics = process_drone_image(
                str(sample_img),
                str(tmp / "raw.png"),
                str(tmp / "enh.png"),
                str(tmp / "heat.png"),
                str(tmp / "mask.png"),
                gamma=1.0,
                sharpen=True,
            )
            self.assertGreater(metrics["canopy_coverage_pct"], 10.0)
            self.assertGreater(metrics["stressed_area_pct"], 1.0)
            self.assertAlmostEqual(
                metrics["canopy_coverage_pct"] + metrics["bare_soil_pct"], 100.0, delta=0.1
            )

            db = Database(tmp / "test.db")
            plot_id = db.get_or_create_plot("Test Plot", "Wheat")
            rec = db.insert_scan(
                {
                    "plot_id": plot_id,
                    "filename": "sample.png",
                    "gamma": 1.0,
                    "otsu_threshold": metrics["otsu_threshold"],
                    "canopy_coverage_pct": metrics["canopy_coverage_pct"],
                    "stressed_area_pct": metrics["stressed_area_pct"],
                    "bare_soil_pct": metrics["bare_soil_pct"],
                    "mean_exg": metrics["mean_exg"],
                    "mean_vari": metrics["mean_vari"],
                    "cluster_count": metrics["cluster_count"],
                    "avg_compactness": metrics["avg_compactness"],
                    "row_uniformity_pct": metrics["row_uniformity_pct"],
                    "raw_path": str(tmp / "raw.png"),
                    "enhanced_path": str(tmp / "enh.png"),
                    "heatmap_path": str(tmp / "heat.png"),
                    "mask_path": str(tmp / "mask.png"),
                }
            )
            self.assertEqual(rec["plot_name"], "Test Plot")
            self.assertEqual(len(db.list_scans()), 1)
            self.assertTrue(db.delete_scan(rec["id"]))

    def test_fastapi_endpoints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            old_db = main_module.db
            old_data_dir = main_module.DATA_DIR
            old_outputs_dir = main_module.OUTPUTS_DIR
            old_uploads_dir = main_module.UPLOADS_DIR
            main_module.DATA_DIR = tmp
            main_module.OUTPUTS_DIR = tmp / "outputs"
            main_module.UPLOADS_DIR = tmp / "uploads"
            main_module.OUTPUTS_DIR.mkdir(parents=True)
            main_module.UPLOADS_DIR.mkdir(parents=True)
            main_module.db = Database(tmp / "test.db")

            try:
                client = TestClient(app)
                res = client.get("/api/health")
                self.assertEqual(res.status_code, 200)
                self.assertEqual(res.json()["status"], "ok")

                res_sample = client.post(
                    "/api/analyze-sample",
                    data={"plot_name": "API Test Plot", "gamma": "1.0"},
                )
                self.assertEqual(res_sample.status_code, 200)
                scan = res_sample.json()["scan"]
                scan_id = scan["id"]
                self.assertIn("canopy_coverage_pct", scan)

                res_csv = client.get(f"/api/scans/{scan_id}/export-csv")
                self.assertEqual(res_csv.status_code, 200)
                self.assertIn("Canopy Coverage (%)", res_csv.text)

                res_del = client.delete(f"/api/scans/{scan_id}")
                self.assertEqual(res_del.status_code, 200)

                oversized = client.post(
                    "/api/analyze",
                    files={"file": ("large.png", b"0" * (10 * 1024 * 1024 + 1), "image/png")},
                )
                self.assertEqual(oversized.status_code, 413)
            finally:
                main_module.db = old_db
                main_module.DATA_DIR = old_data_dir
                main_module.OUTPUTS_DIR = old_outputs_dir
                main_module.UPLOADS_DIR = old_uploads_dir


if __name__ == "__main__":
    unittest.main()
