"""Vercel entrypoint for the packaging classifier."""
import os
from pathlib import Path
from io import BytesIO
import threading, time, warnings
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from flask import Flask, jsonify, request, send_from_directory

ROOT = Path(__file__).resolve().parent
CLASSES = ['cardboard', 'glass', 'metal', 'plastic']
app = Flask(__name__, static_folder=None)
app.config['MAX_CONTENT_LENGTH'] = 4_400_000
MAX_IMAGE_BYTES = 4_000_000
Image.MAX_IMAGE_PIXELS = 20_000_000
_session = None
_init_lock = threading.Lock()

def get_session():
    global _session
    if _session is None:
        with _init_lock:
            if _session is None:
                import onnxruntime as ort
                options = ort.SessionOptions()
                options.intra_op_num_threads = 1
                options.inter_op_num_threads = 1
                _session = ort.InferenceSession(str(ROOT/'model/packaging.onnx'), sess_options=options, providers=['CPUExecutionProvider'])
    return _session

@app.get('/')
def home():
    return send_from_directory(ROOT/'public', 'index.html')

@app.get('/assets/<path:name>')
def assets(name):
    return send_from_directory(ROOT/'public/assets', name)

@app.get('/api/health')
def health():
    try:
        get_session()
        return jsonify(ready=True, classes=CLASSES, model='MobileNetV2', test_accuracy=0.8516483516483516)
    except Exception:
        app.logger.exception('Could not initialize the model')
        return jsonify(ready=False, error='Model unavailable. Check that model/packaging.onnx was committed and included in the deployment.'), 503

@app.post('/api/predict')
def predict():
    upload = request.files.get('image')
    if upload is None:
        return jsonify(error='Choose an image first.'), 400
    raw = upload.read(MAX_IMAGE_BYTES+1)
    if len(raw)>MAX_IMAGE_BYTES:
        return jsonify(error='Choose an image smaller than 4 MB.'), 413
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error',Image.DecompressionBombWarning)
            with Image.open(BytesIO(raw)) as im:
                if im.format not in {'JPEG','PNG','WEBP'}:
                    return jsonify(error='Use a JPG, PNG or WebP image.'), 400
                im.load()
                im=ImageOps.exif_transpose(im).convert('RGB').resize((160,160),Image.Resampling.BILINEAR)
                batch=np.asarray(im,dtype=np.float32)[None]
    except (UnidentifiedImageError,OSError,ValueError,Image.DecompressionBombError,Image.DecompressionBombWarning):
        return jsonify(error='This image could not be read. Try another photo.'), 400
    try:
        t=time.perf_counter();session=get_session()
        scores=session.run(None,{session.get_inputs()[0].name:batch})[0][0]
        if scores.shape!=(4,) or not np.isfinite(scores).all() or (scores<0).any():
            raise ValueError('Invalid scores')
        index=int(np.argmax(scores))
        return jsonify(label=CLASSES[index],confidence=float(scores[index]),probabilities=dict(zip(CLASSES,map(float,scores))),elapsed_ms=round((time.perf_counter()-t)*1000))
    except Exception:
        app.logger.exception('Prediction failed')
        return jsonify(error='Prediction service unavailable. Check the deployment runtime logs.'), 503

@app.errorhandler(413)
def oversized(_):
    return jsonify(error='Choose an image smaller than 4 MB.'), 413

@app.after_request
def headers(response):
    response.headers['X-Content-Type-Options']='nosniff'
    if request.path.startswith('/api/'):
        response.headers['Cache-Control']='no-store'
    return response

if __name__=='__main__':
    app.run(host='0.0.0.0',port=int(os.environ.get('PORT',5000)),debug=False)
