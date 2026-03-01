#!/usr/bin/env python3
"""Industrial pallet + box counting pipeline for Google Colab/Drive.

Pipeline steps:
1) Configure environment and folder structure in Google Drive.
2) Detect pallets with Grounding DINO using semantic prompt "Wooden pallet".
3) Segment boxes inside each pallet ROI with SAM (and optional YOLOv8 refinement).
4) Track pallet IDs and unique boxes to avoid double-counting across frames.
5) Save raw frames, masked frames, random samples, summary JSON, and final video.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

try:
    from google.colab import drive  # type: ignore
except Exception:
    drive = None

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


# ----------------------------- Configuration -----------------------------
@dataclass
class PipelineConfig:
    base_dir: Path = Path(
        "/content/drive/MyDrive/MNA/Proyecto Integrador/Proyecto Final Boxes Counting Version 4.0"
    )
    input_video_name: str = "DJI-VID1.mp4"

    # Industrial thresholds
    pallet_score_threshold: float = 0.35
    pallet_text_threshold: float = 0.25
    pallet_track_iou: float = 0.30
    pallet_max_missing_frames: int = 12

    box_min_area_px: int = 120
    box_max_area_ratio: float = 0.30
    box_match_iou: float = 0.35
    box_centroid_dist_px: float = 45.0

    # Video and runtime
    target_duration_seconds: int = 30
    random_sample_count: int = 3

    # Optional YOLO refinement
    use_yolo_refinement: bool = False
    yolo_model_name: str = "yolov8n.pt"
    yolo_confidence: float = 0.25

    # Models
    dino_model_id: str = "IDEA-Research/grounding-dino-base"
    sam_checkpoint: str = "/content/sam_vit_b_01ec64.pth"
    sam_variant: str = "vit_b"


@dataclass
class Track:
    track_id: int
    bbox_xyxy: np.ndarray
    misses: int = 0


@dataclass
class BoxEntry:
    box_id: int
    bbox_xyxy: np.ndarray
    centroid: Tuple[float, float]


@dataclass
class PalletMemory:
    box_entries: Dict[int, BoxEntry] = field(default_factory=dict)
    final_count: int = 0
    next_box_id: int = 0


# ----------------------------- Utilities -----------------------------
def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def mount_drive_if_available() -> None:
    if drive is None:
        logging.warning("google.colab no está disponible; se asume entorno local ya montado.")
        return
    logging.info("Montando Google Drive...")
    drive.mount("/content/drive", force_remount=False)


def ensure_structure(cfg: PipelineConfig) -> Dict[str, Path]:
    paths = {
        "input_video": cfg.base_dir / "input_video",
        "raw_frames": cfg.base_dir / "raw_frames",
        "masked_frames": cfg.base_dir / "masked_frames",
        "random_samples": cfg.base_dir / "random_samples",
        "json_output": cfg.base_dir / "json_output",
        "final_video": cfg.base_dir / "final_video",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def safe_clear_dir(path: Path, suffixes: Tuple[str, ...] = (".jpg", ".png", ".mp4", ".json")) -> None:
    for item in path.iterdir():
        if item.is_file() and item.suffix.lower() in suffixes:
            item.unlink(missing_ok=True)


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def bbox_centroid(box: np.ndarray) -> Tuple[float, float]:
    return (float((box[0] + box[2]) / 2.0), float((box[1] + box[3]) / 2.0))


def centroid_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def color_from_id(identifier: int) -> Tuple[int, int, int]:
    rng = random.Random(identifier + 12345)
    return (rng.randint(60, 255), rng.randint(60, 255), rng.randint(60, 255))


# ----------------------------- Models -----------------------------
def download_sam_checkpoint_if_needed(cfg: PipelineConfig) -> None:
    checkpoint = Path(cfg.sam_checkpoint)
    if checkpoint.exists():
        return
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    logging.info("Descargando checkpoint SAM en %s", checkpoint)
    os.system(f"wget -q {url} -O {checkpoint}")


def load_models(cfg: PipelineConfig, device: str):
    logging.info("Cargando Grounding DINO: %s", cfg.dino_model_id)
    processor = AutoProcessor.from_pretrained(cfg.dino_model_id)
    dino = AutoModelForZeroShotObjectDetection.from_pretrained(cfg.dino_model_id).to(device)

    download_sam_checkpoint_if_needed(cfg)
    logging.info("Cargando SAM (%s)", cfg.sam_variant)
    sam = sam_model_registry[cfg.sam_variant](checkpoint=cfg.sam_checkpoint)
    sam.to(device)
    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=20,
        pred_iou_thresh=0.85,
        stability_score_thresh=0.90,
        min_mask_region_area=80,
    )

    yolo = None
    if cfg.use_yolo_refinement and YOLO is not None:
        logging.info("Cargando YOLOv8 refinamiento: %s", cfg.yolo_model_name)
        yolo = YOLO(cfg.yolo_model_name)
    elif cfg.use_yolo_refinement:
        logging.warning("YOLOv8 no disponible; se continúa sin refinamiento YOLO.")

    return processor, dino, mask_generator, yolo


# ----------------------------- Detection / Tracking -----------------------------
def detect_pallets(
    frame_bgr: np.ndarray,
    processor,
    dino,
    device: str,
    cfg: PipelineConfig,
) -> List[np.ndarray]:
    image_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    inputs = processor(images=image_pil, text="Wooden pallet.", return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = dino(**inputs)

    target_size = torch.tensor([image_pil.size[::-1]], device=device)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=cfg.pallet_score_threshold,
        text_threshold=cfg.pallet_text_threshold,
        target_sizes=target_size,
    )[0]

    boxes = []
    for box in results["boxes"].detach().cpu().numpy():
        x1, y1, x2, y2 = box.astype(float)
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append(np.array([x1, y1, x2, y2], dtype=np.float32))
    return boxes


def update_pallet_tracks(
    tracks: Dict[int, Track],
    detections: List[np.ndarray],
    next_track_id: int,
    cfg: PipelineConfig,
) -> Tuple[Dict[int, Track], int]:
    unmatched_tracks = set(tracks.keys())
    matched_detection_idx = set()

    # Greedy IoU matching
    pairs = []
    for tid, trk in tracks.items():
        for didx, det in enumerate(detections):
            score = iou_xyxy(trk.bbox_xyxy, det)
            if score >= cfg.pallet_track_iou:
                pairs.append((score, tid, didx))
    pairs.sort(reverse=True, key=lambda x: x[0])

    for score, tid, didx in pairs:
        if tid not in unmatched_tracks or didx in matched_detection_idx:
            continue
        tracks[tid].bbox_xyxy = detections[didx]
        tracks[tid].misses = 0
        unmatched_tracks.discard(tid)
        matched_detection_idx.add(didx)

    # Increase misses for unmatched tracks
    for tid in list(unmatched_tracks):
        tracks[tid].misses += 1
        if tracks[tid].misses > cfg.pallet_max_missing_frames:
            del tracks[tid]

    # Add unmatched detections as new tracks
    for didx, det in enumerate(detections):
        if didx in matched_detection_idx:
            continue
        tracks[next_track_id] = Track(track_id=next_track_id, bbox_xyxy=det, misses=0)
        next_track_id += 1

    return tracks, next_track_id


def detect_boxes_in_pallet_roi(
    frame_bgr: np.ndarray,
    pallet_box_xyxy: np.ndarray,
    mask_generator: SamAutomaticMaskGenerator,
    cfg: PipelineConfig,
    yolo=None,
) -> List[np.ndarray]:
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = pallet_box_xyxy.astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)

    if x2 - x1 < 8 or y2 - y1 < 8:
        return []

    roi = frame_bgr[y1:y2, x1:x2]
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    masks = mask_generator.generate(roi_rgb)
    roi_area = roi.shape[0] * roi.shape[1]

    candidate_boxes: List[np.ndarray] = []
    for m in masks:
        area = int(m.get("area", 0))
        if area < cfg.box_min_area_px:
            continue
        if area > roi_area * cfg.box_max_area_ratio:
            continue

        bx, by, bw, bh = m["bbox"]  # xywh in ROI
        if bw < 5 or bh < 5:
            continue

        gx1, gy1 = x1 + bx, y1 + by
        gx2, gy2 = gx1 + bw, gy1 + bh
        candidate_boxes.append(np.array([gx1, gy1, gx2, gy2], dtype=np.float32))

    if yolo is not None:
        try:
            yolo_results = yolo.predict(roi, conf=cfg.yolo_confidence, verbose=False)
            for pred in yolo_results:
                if pred.boxes is None:
                    continue
                for b in pred.boxes.xyxy.cpu().numpy():
                    gx1, gy1, gx2, gy2 = b
                    candidate_boxes.append(
                        np.array([x1 + gx1, y1 + gy1, x1 + gx2, y1 + gy2], dtype=np.float32)
                    )
        except Exception as exc:
            logging.warning("YOLO refinamiento falló: %s", exc)

    # Deduplicate similar boxes with NMS-like logic
    final_boxes: List[np.ndarray] = []
    for box in sorted(candidate_boxes, key=lambda z: (z[2] - z[0]) * (z[3] - z[1]), reverse=True):
        if any(iou_xyxy(box, kept) > 0.6 for kept in final_boxes):
            continue
        final_boxes.append(box)

    return final_boxes


def update_box_memory(memory: PalletMemory, detected_boxes: List[np.ndarray], cfg: PipelineConfig) -> List[Tuple[int, np.ndarray]]:
    assigned: List[Tuple[int, np.ndarray]] = []
    used_existing = set()

    for box in detected_boxes:
        c = bbox_centroid(box)
        best_id: Optional[int] = None
        best_score = -1.0

        for bid, entry in memory.box_entries.items():
            if bid in used_existing:
                continue
            iou_score = iou_xyxy(box, entry.bbox_xyxy)
            cdist = centroid_distance(c, entry.centroid)
            if iou_score >= cfg.box_match_iou or cdist <= cfg.box_centroid_dist_px:
                score = iou_score - (cdist / 1000.0)
                if score > best_score:
                    best_score = score
                    best_id = bid

        if best_id is None:
            best_id = memory.next_box_id
            memory.next_box_id += 1

        memory.box_entries[best_id] = BoxEntry(box_id=best_id, bbox_xyxy=box, centroid=c)
        used_existing.add(best_id)
        assigned.append((best_id, box))

    memory.final_count = max(memory.final_count, len(memory.box_entries))
    return assigned


# ----------------------------- Visualization -----------------------------
def draw_overlay(
    frame_bgr: np.ndarray,
    pallet_id: int,
    pallet_box: np.ndarray,
    assigned_boxes: List[Tuple[int, np.ndarray]],
) -> np.ndarray:
    output = frame_bgr
    x1, y1, x2, y2 = pallet_box.astype(int)
    cv2.rectangle(output, (x1, y1), (x2, y2), (0, 200, 255), 2)

    for box_id, box in assigned_boxes:
        bx1, by1, bx2, by2 = box.astype(int)
        color = color_from_id((pallet_id * 100000) + box_id)

        # Playground-style semi-transparent fill
        overlay = output.copy()
        cv2.rectangle(overlay, (bx1, by1), (bx2, by2), color, thickness=-1)
        output = cv2.addWeighted(overlay, 0.28, output, 0.72, 0)

        cv2.rectangle(output, (bx1, by1), (bx2, by2), color, 2)
        cv2.putText(
            output,
            f"B{box_id}",
            (bx1, max(14, by1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        output,
        f"Pallet ID: {pallet_id} | Boxes: {len(assigned_boxes)}",
        (x1, max(22, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def add_title(frame_bgr: np.ndarray, title: str) -> np.ndarray:
    out = frame_bgr.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 40), (20, 20, 20), -1)
    cv2.putText(
        out,
        title,
        (12, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


# ----------------------------- Main pipeline -----------------------------
def process_video(cfg: PipelineConfig) -> Dict:
    setup_logging()
    mount_drive_if_available()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info("Dispositivo detectado: %s", device)

    paths = ensure_structure(cfg)
    for key in ["raw_frames", "masked_frames", "random_samples", "json_output", "final_video"]:
        safe_clear_dir(paths[key])

    input_video = paths["input_video"] / cfg.input_video_name
    if not input_video.exists():
        raise FileNotFoundError(f"No se encontró el video de entrada: {input_video}")

    processor, dino, mask_generator, yolo = load_models(cfg, device)

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"No fue posible abrir: {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    max_frames = min(total_frames, int(fps * cfg.target_duration_seconds)) if total_frames else int(fps * cfg.target_duration_seconds)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    logging.info("Procesando hasta %d frames (%.2f FPS, %dx%d)", max_frames, fps, width, height)

    tracks: Dict[int, Track] = {}
    next_track_id = 0
    pallet_memory: Dict[int, PalletMemory] = {}
    processed_frame_paths: List[Path] = []

    frame_idx = 0
    pbar = tqdm(total=max_frames, desc="Procesando video")

    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            raw_path = paths["raw_frames"] / f"frame_{frame_idx:05d}.jpg"
            cv2.imwrite(str(raw_path), frame)

            detections = detect_pallets(frame, processor, dino, device, cfg)
            tracks, next_track_id = update_pallet_tracks(tracks, detections, next_track_id, cfg)

            vis = frame.copy()
            for pallet_id, trk in list(tracks.items()):
                pallet_box = trk.bbox_xyxy.copy()
                detected_boxes = detect_boxes_in_pallet_roi(
                    frame,
                    pallet_box,
                    mask_generator,
                    cfg,
                    yolo=yolo,
                )

                if pallet_id not in pallet_memory:
                    pallet_memory[pallet_id] = PalletMemory()

                assigned_boxes = update_box_memory(pallet_memory[pallet_id], detected_boxes, cfg)
                vis = draw_overlay(vis, pallet_id, pallet_box, assigned_boxes)

            masked_path = paths["masked_frames"] / f"frame_{frame_idx:05d}.jpg"
            cv2.imwrite(str(masked_path), vis)
            processed_frame_paths.append(masked_path)

            if device == "cuda" and frame_idx % 20 == 0:
                torch.cuda.empty_cache()

        except Exception as frame_exc:
            logging.exception("Fallo en frame %d: %s", frame_idx, frame_exc)

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    # Random professional samples
    random_count = min(cfg.random_sample_count, len(processed_frame_paths))
    sampled_paths = random.sample(processed_frame_paths, k=random_count) if random_count > 0 else []
    for idx, source_path in enumerate(sampled_paths):
        image = cv2.imread(str(source_path))
        if image is None:
            continue
        titled = add_title(image, "Industrial Pallet Box Count Visualization")
        cv2.imwrite(str(paths["random_samples"] / f"sample_{idx+1}.jpg"), titled)

    pallets_json = []
    for pallet_id in sorted(pallet_memory.keys()):
        pallets_json.append(
            {
                "pallet_id": int(pallet_id),
                "total_boxes": int(pallet_memory[pallet_id].final_count),
            }
        )

    duration_seconds = min(cfg.target_duration_seconds, frame_idx / fps if fps > 0 else cfg.target_duration_seconds)
    summary = {
        "video_name": cfg.input_video_name,
        "duration_seconds": round(float(duration_seconds), 2),
        "pallets": pallets_json,
        "total_unique_pallets": len(pallets_json),
    }

    json_path = paths["json_output"] / "pallet_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Rebuild final video
    output_video_path = paths["final_video"] / "processed_video.mp4"
    writer = cv2.VideoWriter(
        str(output_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    for p in sorted(processed_frame_paths):
        img = cv2.imread(str(p))
        if img is None:
            continue
        if img.shape[1] != width or img.shape[0] != height:
            img = cv2.resize(img, (width, height))
        writer.write(img)
    writer.release()

    logging.info("JSON guardado en: %s", json_path)
    logging.info("Video final guardado en: %s", output_video_path)

    return summary


def main() -> None:
    cfg = PipelineConfig()
    summary = process_video(cfg)
    print("\nResumen final:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
