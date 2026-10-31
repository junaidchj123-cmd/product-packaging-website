"""Verify and place the exact trained model without requiring a pre-uploaded manifest."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parent
EXPECTED_SHA256 = '372508cb0ddade617ab2e548573f07a7fd9b5719641746ae4427687a25864a4b'
MANIFEST = {
    'sha256': EXPECTED_SHA256,
    'source': 'Trained packaging_model.keras',
    'input_shape': [1, 160, 160, 3],
    'input_dtype': 'float32',
    'input_pixels': 'RGB 0..255; rescaling is embedded',
    'resize': 'Pillow bilinear after EXIF transpose',
    'classes': ['cardboard', 'glass', 'metal', 'plastic'],
    'test_accuracy': 0.8516483516483516,
    'test_images': 182,
}


def main():
    print('PACKAGING BUILD CHECK v2', flush=True)
    destination = ROOT / 'model' / 'packaging.onnx'
    candidates = []
    if destination.is_file():
        candidates.append(destination)
    candidates.extend(
        p for p in sorted(ROOT.rglob('*.onnx'))
        if p not in candidates
        and not any(part in {'.venv', 'node_modules', '.git', '_vendor'}
                    for part in p.relative_to(ROOT).parts)
    )
    matched = None
    for candidate in candidates:
        if hashlib.sha256(candidate.read_bytes()).hexdigest() == EXPECTED_SHA256:
            matched = candidate
            break
    if matched is None:
        if candidates:
            raise RuntimeError(
                'MODEL FILE MISMATCH: the uploaded ONNX file does not match the trained '
                'packaging model. Upload packaging.onnx from Packaging_Model_Folder.zip. '
                'Do not upload the cat/dog model or a Git LFS pointer.'
            )
        raise RuntimeError(
            'MODEL FILE MISSING: no trained packaging.onnx file was found in this '
            'deployment. Upload the actual packaging.onnx file (about 9.6 MB) from '
            'Packaging_Model_Folder.zip into the same GitHub branch Vercel deploys, '
            'then deploy the newest commit. Uploading a ZIP does not extract its contents.'
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if matched.resolve() != destination.resolve():
        shutil.copyfile(matched, destination)
        print('Placed verified model at model/packaging.onnx', flush=True)
    (destination.parent / 'manifest.json').write_text(
        json.dumps(MANIFEST, indent=2), encoding='utf-8'
    )
    (destination.parent / 'class_names.json').write_text(
        json.dumps(MANIFEST['classes'], indent=2), encoding='utf-8'
    )
    required = ['app.py', 'public/index.html', 'public/assets/style.css',
                'public/assets/app.js']
    missing = [name for name in required if not (ROOT / name).is_file()]
    if missing:
        raise RuntimeError(
            'WEBSITE FILES MISSING: ' + ', '.join(missing) + '. Upload these files '
            'with their folder structure into the same repository root as build.py.'
        )
    print('Verified trained model; created manifest and labels; website files present.', flush=True)


if __name__ == '__main__':
    main()
