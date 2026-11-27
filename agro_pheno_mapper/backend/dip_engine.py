"""
Core Digital Image Processing (DIP) Engine for Micro-Plot Drone Agro-Phenotyping.
Grounded in Gonzalez & Woods (4th Edition) across Syllabus Units 2-5:
  - Unit 2 (Spatial Domain): Power-Law (Gamma) transform, Median smoothing, Unsharp masking
  - Unit 3 (Color Processing): Normalized RGB chromaticity, ExG, VARI, HSI Hue/Saturation
  - Unit 4 (Segmentation): Otsu's optimal global thresholding on ExG, Canny boundary overlay
  - Unit 5 (Representation & Description): Morphological opening/closing, Connected Components,
    Area, Compactness (4*pi*A / P^2), and Row Uniformity metrics.
"""

from typing import Dict, Any, Tuple
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def apply_gamma_correction(img_rgb: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Unit 2: Power-law transformation s = c * r^gamma."""
    gamma = float(np.clip(gamma, 0.2, 3.0))
    if abs(gamma - 1.0) < 1e-3:
        return img_rgb.copy()
    normalized = img_rgb.astype(np.float32) / 255.0
    corrected = np.power(normalized, gamma)
    return np.clip(corrected * 255.0, 0, 255).astype(np.uint8)


def apply_spatial_enhancement(
    img_rgb: np.ndarray, gamma: float = 1.0, sharpen: bool = True
) -> np.ndarray:
    """Unit 2: Gamma correction + 3x3 Median filter + optional Unsharp Masking."""
    gamma_img = apply_gamma_correction(img_rgb, gamma=gamma)
    denoised = cv2.medianBlur(gamma_img, 3)
    if not sharpen:
        return denoised
    blurred = cv2.GaussianBlur(denoised, (0, 0), sigmaX=1.2)
    sharpened = cv2.addWeighted(denoised, 1.35, blurred, -0.35, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def compute_vegetation_indices(img_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Unit 3: Color transformations for vegetation phenotyping.
    Returns (ExG, VARI, Hue_deg).
      - ExG = 2g - r - b  (normalized chromaticity)
      - VARI = (G - R) / (G + R - B + eps)
      - Hue (0..360 deg) from Gonzalez & Woods Eq. 6-26
    """
    rgb = img_rgb.astype(np.float32) / 255.0
    R, G, B = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    total = R + G + B + 1e-6
    r, g, b = R / total, G / total, B / total

    exg = 2.0 * g - r - b
    vari = np.clip((G - R) / (G + R - B + 1e-5), -1.0, 1.0)

    # Exact HSI Hue formula (Gonzalez & Woods)
    num = 0.5 * ((R - G) + (R - B))
    den = np.sqrt((R - G) ** 2 + (R - B) * (G - B)) + 1e-6
    theta = np.arccos(np.clip(num / den, -1.0, 1.0))  # radians [0, pi]
    hue_rad = np.where(B <= G, theta, 2.0 * np.pi - theta)
    hue_deg = np.degrees(hue_rad)

    return exg, vari, hue_deg


def otsu_segment_canopy(exg: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Unit 4 & 5: Otsu's optimal global thresholding on ExG + morphological cleanup.
    Returns (binary_canopy_mask_uint8, otsu_threshold_exg_units).
    """
    # Scale ExG from [-1.0, 1.0] to [0, 255] for exact Otsu histogram calculation
    exg_min, exg_max = -0.5, 0.8
    exg_scaled = np.clip((exg - exg_min) / (exg_max - exg_min), 0.0, 1.0)
    exg_u8 = (exg_scaled * 255.0).astype(np.uint8)

    thresh_u8, binary = cv2.threshold(
        exg_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    otsu_exg = exg_min + (thresh_u8 / 255.0) * (exg_max - exg_min)
    if otsu_exg > 0.20:
        otsu_exg = 0.20
        binary = ((exg > otsu_exg).astype(np.uint8)) * 255

    # Unit 5: Morphological Opening (remove soil noise) + Closing (fill leaf holes)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)

    return cleaned, float(otsu_exg)


def classify_stress_and_descriptors(
    canopy_mask: np.ndarray, exg: np.ndarray, hue_deg: np.ndarray
) -> Dict[str, Any]:
    """
    Unit 3, 4 & 5: Segment canopy into Healthy vs. Stressed (chlorotic/yellowing)
    and compute regional shape & row uniformity descriptors.
    """
    total_pixels = canopy_mask.size
    canopy_bool = canopy_mask > 0
    canopy_pixels = int(np.sum(canopy_bool))
    bare_soil_pixels = total_pixels - canopy_pixels

    # Stressed vegetation inside canopy: yellowing/chlorotic hue (25..75 deg) or low ExG
    stressed_bool = canopy_bool & ((hue_deg < 72.0) | (exg < 0.14))
    stressed_pixels = int(np.sum(stressed_bool))
    healthy_pixels = canopy_pixels - stressed_pixels

    canopy_pct = round(100.0 * canopy_pixels / total_pixels, 2)
    stressed_pct = round(100.0 * stressed_pixels / total_pixels, 2)
    healthy_pct = round(100.0 * healthy_pixels / total_pixels, 2)
    bare_soil_pct = round(100.0 * bare_soil_pixels / total_pixels, 2)

    # Connected-component regional descriptors (Unit 5)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(canopy_mask, connectivity=8)
    compactness_vals = []
    valid_clusters = 0

    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < 25:  # ignore tiny sub-leaf specks
            continue
        valid_clusters += 1
        comp_mask = (labels == label_id).astype(np.uint8) * 255
        contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            perim = cv2.arcLength(contours[0], True)
            if perim > 0:
                compactness = min(1.0, float((4.0 * np.pi * area) / (perim ** 2)))
                compactness_vals.append(compactness)

    avg_compactness = round(float(np.mean(compactness_vals)) if compactness_vals else 0.0, 3)

    # Row uniformity: coefficient of variation across vertical/horizontal canopy strips
    col_density = np.mean(canopy_bool, axis=0)
    if np.mean(col_density) > 1e-4:
        cv_ratio = np.std(col_density) / (np.mean(col_density) + 1e-6)
        row_uniformity = round(float(np.clip(100.0 * (1.0 - 0.5 * cv_ratio), 0.0, 100.0)), 2)
    else:
        row_uniformity = 0.0

    return {
        "canopy_coverage_pct": canopy_pct,
        "healthy_area_pct": healthy_pct,
        "stressed_area_pct": stressed_pct,
        "bare_soil_pct": bare_soil_pct,
        "cluster_count": valid_clusters,
        "avg_compactness": avg_compactness,
        "row_uniformity_pct": row_uniformity,
        "stressed_bool": stressed_bool,
    }


def render_exg_heatmap(exg: np.ndarray, out_path: str) -> None:
    """Render false-color ExG vegetation stress heatmap with scientific colorbar."""
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=130)
    im = ax.imshow(exg, cmap="RdYlGn", vmin=-0.35, vmax=0.65)
    ax.set_axis_off()
    cbar = fig.colorbar(im, ax=ax, fraction=0.036, pad=0.02)
    cbar.set_label("Excess Green Index (ExG)", fontsize=9)
    plt.tight_layout(pad=0.2)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def render_mask_with_boundaries(
    canopy_mask: np.ndarray, stressed_bool: np.ndarray, out_path: str
) -> None:
    """Render segmented canopy mask with green boundaries and amber stress highlights."""
    h, w = canopy_mask.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    vis[canopy_mask > 0] = [245, 250, 245]   # crisp white-tint canopy
    vis[stressed_bool] = [245, 175, 45]      # amber-orange stressed zones

    # Unit 4: Canny/Contour boundary outline in vivid green
    contours, _ = cv2.findContours(canopy_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (34, 197, 94), 1)
    cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))


def process_drone_image(
    input_path: str,
    raw_out: str,
    enhanced_out: str,
    heatmap_out: str,
    mask_out: str,
    gamma: float = 1.0,
    sharpen: bool = True,
) -> Dict[str, Any]:
    """Run complete end-to-end DIP pipeline and save all 4 visual stages."""
    bgr = cv2.imread(input_path)
    if bgr is None:
        raise ValueError(f"Could not decode image at {input_path}")

    # Resize very large drone frames to max 1024px wide for snappy interactive speed
    h, w = bgr.shape[:2]
    if w > 1024:
        scale = 1024.0 / w
        bgr = cv2.resize(bgr, (1024, int(h * scale)), interpolation=cv2.INTER_AREA)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    cv2.imwrite(raw_out, bgr)

    # Unit 2: Spatial enhancement
    enhanced_rgb = apply_spatial_enhancement(rgb, gamma=gamma, sharpen=sharpen)
    cv2.imwrite(enhanced_out, cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2BGR))

    # Unit 3: Color indices & HSI
    exg, vari, hue_deg = compute_vegetation_indices(enhanced_rgb)

    # Unit 4: Otsu segmentation
    canopy_mask, otsu_thresh = otsu_segment_canopy(exg)

    # Unit 5: Descriptors & stress classification
    metrics = classify_stress_and_descriptors(canopy_mask, exg, hue_deg)

    # Render visual outputs
    render_exg_heatmap(exg, heatmap_out)
    render_mask_with_boundaries(canopy_mask, metrics.pop("stressed_bool"), mask_out)

    metrics["otsu_threshold"] = round(otsu_thresh, 4)
    metrics["mean_exg"] = round(float(np.mean(exg)), 4)
    metrics["mean_vari"] = round(float(np.mean(vari)), 4)
    return metrics


def generate_sample_drone_plot(out_path: str, width: int = 720, height: int = 480) -> str:
    """
    Generate a realistic synthetic top-down drone crop plot image (soil, healthy green rows,
    chlorotic yellow stress patches, and subtle cloud shadow) for instant demoing.
    """
    rng = np.random.default_rng(42)
    img = np.zeros((height, width, 3), dtype=np.float32)

    # Textured brown soil background
    img[:, :, 0] = 108 + rng.normal(0, 10, (height, width))
    img[:, :, 1] = 84 + rng.normal(0, 8, (height, width))
    img[:, :, 2] = 64 + rng.normal(0, 7, (height, width))

    yy, xx = np.mgrid[0:height, 0:width]

    # Tilted crop rows (8 parallel rows)
    row_coord = (xx * 0.92 + yy * 0.39)
    for r_idx in range(1, 11):
        center = r_idx * 74
        for offset in range(18, height + width, 26):
            # Individual plant canopy clusters along the row
            px = int((center - offset * 0.39) / 0.92) + int(rng.integers(-3, 4))
            py = int(offset * 0.92) - 40
            if -20 <= px < width + 20 and -20 <= py < height + 20:
                rad = int(rng.integers(14, 23))
                dist2 = (xx - px) ** 2 + (yy - py) ** 2
                leaf_mask = dist2 <= (rad ** 2)
                # Stress zone in middle-left rows
                is_stressed = (220 < px < 420 and 140 < py < 330)
                if is_stressed:
                    img[leaf_mask, 0] = 172 + rng.normal(0, 10, np.sum(leaf_mask))
                    img[leaf_mask, 1] = 186 + rng.normal(0, 10, np.sum(leaf_mask))
                    img[leaf_mask, 2] = 52 + rng.normal(0, 8, np.sum(leaf_mask))
                else:
                    img[leaf_mask, 0] = 54 + rng.normal(0, 10, np.sum(leaf_mask))
                    img[leaf_mask, 1] = 148 + rng.normal(0, 14, np.sum(leaf_mask))
                    img[leaf_mask, 2] = 44 + rng.normal(0, 9, np.sum(leaf_mask))

    # Subtle cloud shadow gradient on right edge
    shadow = 1.0 - 0.22 * np.clip((xx - width * 0.65) / (width * 0.35), 0.0, 1.0)
    img *= shadow[:, :, None]

    uint8_rgb = np.clip(img, 0, 255).astype(np.uint8)
    cv2.imwrite(out_path, cv2.cvtColor(uint8_rgb, cv2.COLOR_RGB2BGR))
    return out_path
