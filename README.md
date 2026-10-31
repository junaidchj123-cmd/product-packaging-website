# Product Packaging Classification Website

Upload an image to classify it as cardboard, glass, metal or plastic. Includes a responsive HTML/CSS/JavaScript frontend, Flask prediction API, the actual trained ONNX model, sample photos and evaluation charts.

See **DEPLOY_VERCEL.txt** for step-by-step GitHub and Vercel instructions.

## Files and runtime
- `app.py`: Vercel-compatible Flask entrypoint with `/`, `/api/health` and `/api/predict`.
- `public/index.html`: complete homepage.
- `public/assets/`: stylesheet, JavaScript, examples and training-result images.
- `model/packaging.onnx`: trained model, included in this package. No build-time model download.
- `model/manifest.json`: SHA-256, input specification and class order.
- `build.py`: verifies model integrity before deployment.
- `requirements.txt` and `pyproject.toml`: matching pinned runtime dependencies; Python 3.12.
- `vercel.json`: Flask preset, function duration and model inclusion.

TensorFlow, Keras and training dependencies are not needed at runtime. The local installed runtime dependencies plus application total approximately 169 MiB; the actual Vercel bundle can differ. No model weights are downloaded during inference.

## Prediction contract
`POST /api/predict` accepts multipart field `image`, JPG/PNG/WebP, up to 4,000,000 bytes. The server applies EXIF orientation, RGB conversion and Pillow bilinear resizing to 160 x 160. It passes float32 pixels in 0..255 into ONNX. Rescaling to -1..1 is inside the model. Output contains label, confidence, all four class scores and elapsed time. Images are processed in memory and are not saved by the app.

Class order: cardboard, glass, metal, plastic. These are material labels, not package shapes. The model always chooses a class even for unrelated inputs. Scores are not calibrated probabilities.

## Actual model results
Trained on the uploaded 1,790-image four-material TrashNet subset: 1,431 train, 177 validation, 182 test. Test accuracy 85.1648% (155/182). The backbone started from the earlier ImageNet-pretrained and cat/dog-fine-tuned MobileNetV2 model; the binary head was replaced and the packaging model trained and fine-tuned. This is not a fresh untouched ImageNet initialization.

ONNX conversion was checked against the saved Keras model for ALL 182 test images: 182 identical class predictions, maximum absolute class-score difference 0.00001193. Test accuracy remains 155/182. The conversion does not retrain the model. Small numerical differences may occur across runtime versions.

Checks completed: homepage and asset routes, model health, all test-image predictions, missing/invalid/oversized uploads, model-path exclusion. Browser QA covered the frontend layout and sample selection. The cloud preview only served static files, so browser-to-backend prediction was checked via local Flask API tests rather than in that preview. No live Vercel deployment was performed here.

## Source and attribution
Dataset and sample photos: TrashNet by Gary Thung and Mindy Yang, https://github.com/garythung/trashnet. Review the original source terms before redistribution. The source license, when included, is retained as DATASET_SOURCE_LICENSE. Training results were generated from the uploaded project dataset.

Vercel references consulted:
- https://vercel.com/docs/frameworks/backend/flask
- https://vercel.com/docs/functions/runtimes/python
- https://vercel.com/docs/functions/runtimes/python/python-version

By Junaid Shahid, Deep Learning Semester Project.
