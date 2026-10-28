"""Train the DeepFashion2 three-class garment classifier.

The default paths target the local dataset supplied for this project:

    python backend/scripts/train_classifier.py

For a quick verification run:

    python backend/scripts/train_classifier.py --max-train-samples 256 \
        --max-val-samples 128 --epochs 1 --batch-size 16
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models, transforms

from app.dataset import DeepFashion2Dataset

LABELS = ("bottom", "fullwear", "top")
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=r"E:\DeepFashion2")
    parser.add_argument("--output", default="garment_classifier.pt")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def make_dataset(root: Path, split: str, transform, max_samples: int | None):
    split_root = root / "deepfashion2_original_images" / split
    return DeepFashion2Dataset(
        split_root / "image",
        split_root / "annos",
        transform=transform,
        max_samples=max_samples,
    )


def build_model(pretrained: bool) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(LABELS))
    return model


def evaluate(model, loader, device) -> tuple[float, float, list[list[int]]]:
    model.eval()
    correct = total = 0
    loss_total = 0.0
    confusion = [[0 for _ in LABELS] for _ in LABELS]
    loss_fn = nn.CrossEntropyLoss()
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device))
            labels = labels.to(device)
            loss_total += loss_fn(logits, labels).item() * labels.size(0)
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            for actual, predicted in zip(labels.cpu().tolist(), predictions.cpu().tolist()):
                confusion[actual][predicted] += 1
    return loss_total / max(total, 1), correct / max(total, 1), confusion


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    root = Path(args.dataset_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    validation_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    train_set = make_dataset(root, "train", train_transform, args.max_train_samples)
    validation_set = make_dataset(root, "validation", validation_transform, args.max_val_samples)
    if not train_set or not validation_set:
        raise RuntimeError("DeepFashion2 train/validation samples were not found.")

    labels = [label for _, label, _ in train_set.samples]
    counts = {label: labels.count(label) for label in LABELS}
    weights = [1.0 / counts[label] for label in labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.workers, pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == "cuda",
    )
    model = build_model(not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss()
    best_accuracy = -1.0
    output_path = Path(args.output)
    print(f"device={device} train={len(train_set)} validation={len(validation_set)} counts={counts}")

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_total = 0
        for images, labels_batch in train_loader:
            images, labels_batch = images.to(device), labels_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(images), labels_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * labels_batch.size(0)
            train_total += labels_batch.size(0)
        scheduler.step()
        validation_loss, validation_accuracy, confusion = evaluate(
            model, validation_loader, device
        )
        print(
            f"epoch={epoch + 1}/{args.epochs} "
            f"train_loss={train_loss / train_total:.4f} "
            f"val_loss={validation_loss:.4f} val_accuracy={validation_accuracy:.4f}"
        )
        if validation_accuracy >= best_accuracy:
            best_accuracy = validation_accuracy
            torch.save({
                "state_dict": model.cpu().state_dict(),
                "labels": LABELS,
                "mean": MEAN,
                "std": STD,
                "architecture": "resnet18",
                "validation_accuracy": validation_accuracy,
                "confusion_matrix": confusion,
                "dataset_root": str(root),
            }, output_path)
            model.to(device)

    metrics_path = output_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps({
        "labels": LABELS,
        "best_validation_accuracy": best_accuracy,
        "train_samples": len(train_set),
        "validation_samples": len(validation_set),
        "checkpoint": str(output_path),
    }, indent=2), encoding="utf-8")
    print(f"saved={output_path} metrics={metrics_path}")


if __name__ == "__main__":
    main()
