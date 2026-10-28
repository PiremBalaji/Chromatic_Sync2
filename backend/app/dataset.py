import json
from itertools import islice
from pathlib import Path
from typing import Iterator

from PIL import Image
from torch.utils.data import Dataset

LABELS = {
    1: "top", 2: "top", 3: "top", 4: "top", 5: "top", 6: "top",
    7: "bottom", 8: "bottom", 9: "bottom",
    10: "fullwear", 11: "fullwear", 12: "fullwear",
    13: "fullwear", 14: "fullwear", 15: "fullwear",
}


def iter_deepfashion2_samples(
    image_dir: str | Path, annotation_dir: str | Path
) -> Iterator[tuple[Path, str, tuple[int, int, int, int]]]:
    """Yield cropped garment image paths and grouped labels from DF2 annotations."""
    for annotation_path in sorted(Path(annotation_dir).glob("*.json")):
        data = json.loads(annotation_path.read_text(encoding="utf-8"))
        image_path = Path(image_dir) / f"{annotation_path.stem}.jpg"
        items = (
            value for key, value in data.items()
            if key.startswith("item") and isinstance(value, dict)
        )
        for item in items:
            category = LABELS.get(item.get("category_id"))
            box = item.get("bounding_box")
            if category and box and len(box) == 4 and image_path.exists():
                left, top, right, bottom = (int(value) for value in box)
                if right > left and bottom > top:
                    yield image_path, category, (left, top, right, bottom)


class DeepFashion2Dataset(Dataset):
    """DeepFashion2 garment crops grouped into top, bottom, and fullwear."""

    def __init__(
        self,
        image_dir: str,
        annotation_dir: str,
        transform=None,
        max_samples: int | None = None,
    ):
        samples = iter_deepfashion2_samples(image_dir, annotation_dir)
        self.samples = list(samples if max_samples is None else islice(samples, max_samples))
        self.transform = transform
        self.label_to_id = {"bottom": 0, "fullwear": 1, "top": 2}
        self.id_to_label = {value: key for key, value in self.label_to_id.items()}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label, box = self.samples[index]
        image = Image.open(image_path).convert("RGB").crop(box)
        if self.transform:
            image = self.transform(image)
        return image, self.label_to_id[label]
