"""Fail the build if the trained model is missing or truncated."""
from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parent
manifest=json.loads((root/'model/manifest.json').read_text())
path=root/'model/packaging.onnx'
if not path.is_file():
    raise RuntimeError('Missing model/packaging.onnx. Upload the model folder, not only the HTML files.')
if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['sha256']:
    raise RuntimeError('Model file is incomplete or changed. Upload model/packaging.onnx from the provided ZIP again.')
for name in ['index.html','assets/style.css','assets/app.js']:
    if not (root/'public'/name).is_file():raise RuntimeError('Missing public/'+name)
print('Verified trained packaging model and website assets.')
