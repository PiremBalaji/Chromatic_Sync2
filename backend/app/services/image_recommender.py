import hashlib
import io
import json
import os
from pathlib import Path
from threading import Lock

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image

from app.dataset import LABELS

DATASET_ROOT = Path(os.getenv("DEEPFASHION2_ROOT", r"E:\DeepFashion2"))
MAX_CANDIDATES_PER_CATEGORY = 36
_catalog: dict[str, tuple[Path, tuple[int, int, int, int], str]] = {}
_catalog_by_category: dict[str, list[str]] = {label: [] for label in ("top", "bottom", "fullwear")}
_catalog_lock = Lock()


def _ensure_catalog() -> None:
    if any(_catalog_by_category.values()):
        return
    with _catalog_lock:
        if any(_catalog_by_category.values()):
            return
        for split in ("train", "validation"):
            image_dir = DATASET_ROOT / "deepfashion2_original_images" / split / "image"
            annotation_dir = DATASET_ROOT / "deepfashion2_original_images" / split / "annos"
            for annotation_path in sorted(annotation_dir.glob("*.json")):
                image_path = image_dir / f"{annotation_path.stem}.jpg"
                if not image_path.exists():
                    continue
                try:
                    data = json.loads(annotation_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                for item_key, item in data.items():
                    if not item_key.startswith("item") or not isinstance(item, dict):
                        continue
                    category = LABELS.get(item.get("category_id"))
                    box = item.get("bounding_box")
                    if category not in _catalog_by_category or not isinstance(box, list) or len(box) != 4:
                        continue
                    bounds = tuple(int(value) for value in box)
                    if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
                        continue
                    sample_id = hashlib.sha1(
                        f"{split}:{annotation_path.stem}:{item_key}".encode()
                    ).hexdigest()[:16]
                    _catalog[sample_id] = (image_path, bounds, category)
                    _catalog_by_category[category].append(sample_id)
                if all(len(items) >= MAX_CANDIDATES_PER_CATEGORY for items in _catalog_by_category.values()):
                    return


def recommendation_images(category: str, color: str, count: int = 3) -> list[dict[str, str]]:
    _ensure_catalog()
    target = "top" if category in {"bottom", "fullwear"} else "bottom"
    candidates = _catalog_by_category[target]
    if not candidates:
        return []
    start = int(hashlib.sha1(f"{category}:{color}".encode()).hexdigest(), 16) % len(candidates)
    selected = [candidates[(start + index) % len(candidates)] for index in range(min(count, len(candidates)))]
    return [
        {
            "image_url": f"/api/recommendations/images/{sample_id}",
            "category": _catalog[sample_id][2],
        }
        for sample_id in selected
    ]


def image_response(sample_id: str) -> StreamingResponse:
    _ensure_catalog()
    sample = _catalog.get(sample_id)
    if sample is None:
        raise HTTPException(status_code=404, detail="Recommendation image not found.")
    image_path, bounds, _ = sample
    try:
        with Image.open(image_path) as image:
            crop = image.convert("RGB").crop(bounds)
            output = io.BytesIO()
            crop.save(output, format="JPEG", quality=88)
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=404, detail="Recommendation image is unavailable.") from error
    output.seek(0)
    return StreamingResponse(output, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=3600"})
