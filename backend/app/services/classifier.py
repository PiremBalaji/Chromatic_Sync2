import os
from pathlib import Path

from PIL import Image

LABELS = ("bottom", "fullwear", "top")
MODEL_NAME = "deepfashion2-resnet18"


class GarmentClassifier:
    def __init__(self) -> None:
        self._model = None
        self._transform = None
        self.model_path = Path(os.getenv("GARMENT_MODEL_PATH", "garment_classifier.pt"))
        if self.model_path.exists():
            self._load_model()

    @property
    def available(self) -> bool:
        return self._model is not None

    def _load_model(self) -> None:
        import torch
        from torchvision import models, transforms

        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, len(LABELS))
        checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=True)
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state_dict)
        model.eval()
        self._model = model
        self.model_name = MODEL_NAME
        self._transform = transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )

    def predict(self, image: Image.Image) -> tuple[str, float, str]:
        if not self.available:
            raise RuntimeError("No trained garment checkpoint is available.")
        import torch

        assert self._transform is not None
        assert self._model is not None
        tensor = self._transform(image.convert("RGB")).unsqueeze(0)
        with torch.inference_mode():
            probabilities = torch.softmax(self._model(tensor), dim=1)[0]
        confidence, index = torch.max(probabilities, dim=0)
        return LABELS[index.item()], confidence.item(), MODEL_NAME


classifier = GarmentClassifier()
