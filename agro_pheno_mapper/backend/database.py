"""
Separate SQLite Database Layer for AgroPheno Mapper.
Manages field plots, scan records, DIP parameters, and computed vegetation metrics.
"""

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional
from contextlib import contextmanager

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "agropheno.db"


class Database:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS plots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    crop_type TEXT NOT NULL DEFAULT 'Row Crop',
                    location TEXT DEFAULT '',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS scans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plot_id INTEGER NOT NULL,
                    filename TEXT NOT NULL,
                    gamma REAL NOT NULL DEFAULT 1.0,
                    otsu_threshold REAL NOT NULL,
                    canopy_coverage_pct REAL NOT NULL,
                    stressed_area_pct REAL NOT NULL,
                    bare_soil_pct REAL NOT NULL,
                    mean_exg REAL NOT NULL,
                    mean_vari REAL NOT NULL,
                    cluster_count INTEGER NOT NULL,
                    avg_compactness REAL NOT NULL,
                    row_uniformity_pct REAL NOT NULL,
                    raw_path TEXT NOT NULL,
                    enhanced_path TEXT NOT NULL,
                    heatmap_path TEXT NOT NULL,
                    mask_path TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (plot_id) REFERENCES plots(id) ON DELETE CASCADE
                );
                """
            )
            # Seed a default plot if empty
            cur = conn.execute("SELECT COUNT(*) AS cnt FROM plots")
            if cur.fetchone()["cnt"] == 0:
                conn.execute(
                    "INSERT INTO plots (name, crop_type, location) VALUES (?, ?, ?)",
                    ("Plot A-1 (Demo)", "Maize / Row Crop", "Experimental Block North"),
                )

    # --- Plot CRUD ---
    def list_plots(self) -> List[Dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT id, name, crop_type, location, created_at FROM plots ORDER BY id DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    def create_plot(self, name: str, crop_type: str = "Row Crop", location: str = "") -> Dict[str, Any]:
        with self.connect() as conn:
            cur = conn.execute(
                "INSERT INTO plots (name, crop_type, location) VALUES (?, ?, ?)",
                (name.strip(), crop_type.strip(), location.strip()),
            )
            row = conn.execute("SELECT * FROM plots WHERE id = ?", (cur.lastrowid,)).fetchone()
            return dict(row)

    def get_or_create_plot(self, name: str, crop_type: str = "Row Crop") -> int:
        with self.connect() as conn:
            row = conn.execute("SELECT id FROM plots WHERE name = ?", (name.strip(),)).fetchone()
            if row:
                return int(row["id"])
            cur = conn.execute(
                "INSERT INTO plots (name, crop_type) VALUES (?, ?)",
                (name.strip(), crop_type.strip()),
            )
            return int(cur.lastrowid)

    # --- Scan CRUD ---
    def insert_scan(self, record: Dict[str, Any]) -> Dict[str, Any]:
        with self.connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO scans (
                    plot_id, filename, gamma, otsu_threshold,
                    canopy_coverage_pct, stressed_area_pct, bare_soil_pct,
                    mean_exg, mean_vari, cluster_count, avg_compactness,
                    row_uniformity_pct, raw_path, enhanced_path, heatmap_path, mask_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["plot_id"],
                    record["filename"],
                    record["gamma"],
                    record["otsu_threshold"],
                    record["canopy_coverage_pct"],
                    record["stressed_area_pct"],
                    record["bare_soil_pct"],
                    record["mean_exg"],
                    record["mean_vari"],
                    record["cluster_count"],
                    record["avg_compactness"],
                    record["row_uniformity_pct"],
                    record["raw_path"],
                    record["enhanced_path"],
                    record["heatmap_path"],
                    record["mask_path"],
                ),
            )
            row = conn.execute(
                """
                SELECT s.*, p.name AS plot_name, p.crop_type
                FROM scans s
                JOIN plots p ON s.plot_id = p.id
                WHERE s.id = ?
                """,
                (cur.lastrowid,),
            ).fetchone()
            return dict(row)

    def list_scans(self, plot_id: Optional[int] = None, limit: int = 50) -> List[Dict[str, Any]]:
        with self.connect() as conn:
            if plot_id is not None:
                rows = conn.execute(
                    """
                    SELECT s.*, p.name AS plot_name, p.crop_type
                    FROM scans s
                    JOIN plots p ON s.plot_id = p.id
                    WHERE s.plot_id = ?
                    ORDER BY s.id DESC LIMIT ?
                    """,
                    (plot_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT s.*, p.name AS plot_name, p.crop_type
                    FROM scans s
                    JOIN plots p ON s.plot_id = p.id
                    ORDER BY s.id DESC LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
            return [dict(r) for r in rows]

    def get_scan(self, scan_id: int) -> Optional[Dict[str, Any]]:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT s.*, p.name AS plot_name, p.crop_type
                FROM scans s
                JOIN plots p ON s.plot_id = p.id
                WHERE s.id = ?
                """,
                (scan_id,),
            ).fetchone()
            return dict(row) if row else None

    def delete_scan(self, scan_id: int) -> bool:
        with self.connect() as conn:
            cur = conn.execute("DELETE FROM scans WHERE id = ?", (scan_id,))
            return cur.rowcount > 0
