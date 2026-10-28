import base64
import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from app.services.classifier import classifier
from app.services.image_recommender import recommendation_images
from app.services.recommender import recommendations_for

COLOR_NAMES = {
    "black": (25, 25, 25),
    "white": (235, 235, 235),
    "gray": (128, 128, 128),
    "red": (190, 45, 45),
    "orange": (220, 125, 35),
    "yellow": (220, 190, 35),
    "green": (55, 145, 75),
    "blue": (55, 100, 190),
    "purple": (130, 75, 165),
    "pink": (215, 105, 145),
    "brown": (130, 80, 45),
    "beige": (205, 180, 135),
}

# DeepFashion2 category IDs grouped into the product types needed by the app.
DEEPFASHION2_GROUPS = {
    "top": {1, 2, 3, 4, 5, 6},
    "bottom": {7, 8, 9},
    "fullwear": {10, 11, 12, 13, 14, 15},
}


def _grabcut_result(image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.asarray(image.convert("RGB").resize((160, 160)), dtype=np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    mask = np.full((160, 160), cv2.GC_PR_BGD, dtype=np.uint8)
    mask[8:-8, 8:-8] = cv2.GC_PR_FGD
    background_model = np.zeros((1, 65), dtype=np.float64)
    foreground_model = np.zeros((1, 65), dtype=np.float64)
    cv2.grabCut(
        bgr,
        mask,
        (8, 8, 144, 144),
        background_model,
        foreground_model,
        5,
        cv2.GC_INIT_WITH_RECT,
    )
    foreground = np.logical_or(mask == cv2.GC_FGD, mask == cv2.GC_PR_FGD)
    if int(foreground.sum()) < 100:
        raise ValueError("GrabCut did not isolate enough garment pixels.")
    return rgb, foreground


def _grabcut_foreground(image: Image.Image) -> np.ndarray:
    _, foreground = _grabcut_result(image)
    rgb = np.asarray(image.convert("RGB").resize((160, 160)), dtype=np.uint8)
    return rgb[foreground]


def _grabcut_preview(image: Image.Image) -> str:
    rgb, foreground = _grabcut_result(image)
    preview = np.zeros((160, 160, 4), dtype=np.uint8)
    preview[:, :, :3] = rgb
    preview[:, :, 3] = np.where(foreground, 255, 0).astype(np.uint8)
    output = io.BytesIO()
    Image.fromarray(preview, mode="RGBA").save(output, format="PNG")
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _dominant_color(image: Image.Image) -> tuple[str, str]:
    try:
        foreground = _grabcut_foreground(image)
    except (cv2.error, ValueError):
        resized = np.asarray(image.convert("RGB").resize((96, 96)), dtype=np.uint8)
        foreground = resized[12:84, 16:80].reshape(-1, 3)

    samples = np.float32(foreground)
    cluster_count = min(3, len(samples))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, labels, centers = cv2.kmeans(
        samples,
        cluster_count,
        None,
        criteria,
        5,
        cv2.KMEANS_PP_CENTERS,
    )
    counts = np.bincount(labels.ravel(), minlength=cluster_count)
    rgb = tuple(int(round(value)) for value in centers[int(np.argmax(counts))])
    name = min(
        COLOR_NAMES,
        key=lambda candidate: sum(
            (rgb[index] - COLOR_NAMES[candidate][index]) ** 2 for index in range(3)
        ),
    )
    return name, "#{:02x}{:02x}{:02x}".format(*rgb)


def _baseline_category(image: Image.Image) -> tuple[str, float]:
    """Return a useful baseline until a trained DeepFashion2 model is installed.

    The inference interface is intentionally kept independent from this fallback,
    so a checkpoint can replace it without changing the API or frontend.
    """
    width, height = image.size
    ratio = width / max(height, 1)
    if ratio > 1.35:
        return "bottom", 0.58
    if ratio < 0.68:
        return "fullwear", 0.54
    return "top", 0.56


def analyze_garment(image_path: str) -> dict:
    try:
        image = Image.open(image_path)
        image.load()
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError("The uploaded file is not a readable image.") from error

    if classifier.available:
        category, confidence, model_name = classifier.predict(image)
    else:
        category, confidence = _baseline_category(image)
        model_name = "baseline-v0"
    try:
        grabcut_image = _grabcut_preview(image)
    except (cv2.error, ValueError):
        grabcut_image = None
    color, hex_value = _dominant_color(image)
    recommendations = recommendations_for(category, color)
    images = recommendation_images(category, color, len(recommendations))
    for recommendation, image in zip(recommendations, images):
        recommendation.update(image)
    return {
        "category": category,
        "confidence": confidence,
        "color": {"name": color, "hex": hex_value},
        "grabcut_image": grabcut_image,
        "recommendations": recommendations,
        "model": model_name,
        "filename": Path(image_path).name,
    }
