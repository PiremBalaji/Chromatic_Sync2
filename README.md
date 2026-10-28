# Chromatic Sync

Chromatic Sync analyzes an uploaded garment image and returns:

- a grouped garment type: `top`, `bottom`, or `fullwear`
- a foreground-aware dominant color and hex value
- compatible outfit suggestions (tops for bottoms, bottoms for tops, and styling for fullwear)
- compatible garment crop images selected from the local DeepFashion2 dataset

The current `baseline-v0` analyzer is intentionally runnable without a model checkpoint. It uses image geometry for a rough category estimate and foreground pixel quantization for color. If `garment_classifier.pt` exists in the working directory, the API automatically loads it and reports `deepfashion2-resnet18` instead. The DeepFashion2 dataset adapter and training scaffold are included so the baseline can be replaced with a trained classifier.

Color extraction first segments the garment with OpenCV GrabCut, then uses OpenCV K-means clustering over the segmented garment pixels. Model retraining is not required when this color logic changes.

## Run locally

### Backend

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
$env:PYTHONPATH = "backend"
uvicorn app.main:app --reload --app-dir backend
```

### Frontend

```powershell
cd frontend
npm install
npm run dev
```

Open the Vite URL (usually `http://localhost:5173`). In development, Vite proxies `/api` requests to the FastAPI server on port 8000. To use a separately hosted API, set `VITE_API_URL`.

## DeepFashion2 model training

DeepFashion2 annotations contain category IDs from 1 to 15. The adapter maps IDs 1–6 to `top`, 7–9 to `bottom`, and 10–15 to `fullwear`. The project is configured for your dataset at `E:\DeepFashion2`. The real training entrypoint is:

```powershell
python backend\scripts\train_classifier.py
```

It reads:

```text
E:\DeepFashion2\deepfashion2_original_images\train\image
E:\DeepFashion2\deepfashion2_original_images\train\annos
E:\DeepFashion2\deepfashion2_original_images\validation\image
E:\DeepFashion2\deepfashion2_original_images\validation\annos
```

The training system:

- crops each annotated garment bounding box
- maps DeepFashion2 IDs 1–6 to `top`, 7–9 to `bottom`, and 10–15 to `fullwear`
- applies augmentation and ImageNet normalization
- uses balanced sampling for the grouped classes
- evaluates validation loss, accuracy, and confusion matrix every epoch
- saves the best ResNet18 checkpoint and a metrics JSON file

For a quick local verification before a full run:

```powershell
python backend\scripts\train_classifier.py --max-train-samples 256 --max-val-samples 128 --epochs 1 --batch-size 16 --no-pretrained
```

For a stronger run with ImageNet initialization:

```powershell
python backend\scripts\train_classifier.py --epochs 8 --batch-size 32
```

The generated `garment_classifier.pt` is loaded automatically by the API on restart. The UI then reports `deepfashion2-resnet18` instead of `baseline-v0`.

## Image recommendations

The API uses the same DeepFashion2 annotations to select compatible garment crops:

- uploaded top -> bottom crops
- uploaded bottom -> top crops
- uploaded fullwear -> top layering crops

The dataset remains outside the web root. The API exposes only validated crop IDs through `/api/recommendations/images/{sample_id}`.

## API

`POST /api/analyze` accepts a multipart field named `file` and returns:

```json
{
  "category": "top",
  "confidence": 0.56,
  "color": {"name": "blue", "hex": "#1050b0"},
  "recommendations": [
    {"type": "bottom", "item": "High-waisted trousers", "color": "white", "reason": "Balances the silhouette"}
  ],
  "model": "baseline-v0"
}
```
