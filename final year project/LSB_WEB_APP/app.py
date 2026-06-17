"""
Unified LSB Watermarking — Flask Application
Supports:
  • Image-to-Image watermarking  →  POST /process
  • Audio-to-Image watermarking  →  POST /process-audio  (1, 4, 7-bit auto)
"""

from flask import Flask, render_template, request, send_from_directory
import os
from PIL import Image
import numpy as np

from utils.image_watermark import embed_extract
from utils.image_watermark_1bit import embed_extract_1bit_rgb
from utils.audio_watermark import process_audio_watermark
from utils.metrics import mse, psnr

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Flask(__name__)

UPLOAD_DIR = os.path.join("static", "uploads")
OUTPUT_DIR = os.path.join("static", "outputs")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html", active_tab="image")


# ── Image Watermarking ──────────────────────────────────────────────────────

@app.route("/process", methods=["POST"])
def process_image():
    cover_file = request.files.get("cover")
    watermark_file = request.files.get("watermark")
    lsb_method = request.form.get("lsb_method", "4bit")

    if not cover_file or not watermark_file:
        return render_template(
            "index.html",
            error="Please upload both a cover image and a watermark image.",
            active_tab="image",
        )

    cover_path = os.path.join(UPLOAD_DIR, cover_file.filename)
    watermark_path = os.path.join(UPLOAD_DIR, watermark_file.filename)
    cover_file.save(cover_path)
    watermark_file.save(watermark_path)

    try:
        if lsb_method == "1bit":
            watermarked_path, ext_wm_path, ext_cover_path = embed_extract_1bit_rgb(
                cover_path, watermark_path, OUTPUT_DIR
            )
            method_name = "1-bit LSB (per RGB channel)"
        else:
            watermarked_path, ext_wm_path, ext_cover_path = embed_extract(
                cover_path, watermark_path, OUTPUT_DIR
            )
            method_name = "4-bit LSB"
    except Exception as exc:
        return render_template(
            "index.html",
            error=f"Processing failed: {exc}",
            active_tab="image",
        )

    # Quality metrics
    cover_gray = np.array(Image.open(cover_path).convert("L"))
    wm_gray = np.array(Image.open(watermarked_path).convert("L"))
    ext_cover_gray = np.array(Image.open(ext_cover_path).convert("L"))
    ext_wm_gray = np.array(Image.open(ext_wm_path).convert("L"))

    result = {
        "watermarked": watermarked_path.replace("\\", "/"),
        "extracted_cover": ext_cover_path.replace("\\", "/"),
        "extracted_watermarked": ext_wm_path.replace("\\", "/"),
        "mse1": round(mse(ext_wm_gray, wm_gray), 4),
        "psnr1": round(psnr(ext_wm_gray, wm_gray), 2),
        "mse2": round(mse(ext_cover_gray, cover_gray), 4),
        "psnr2": round(psnr(ext_cover_gray, cover_gray), 2),
        "method": method_name,
    }

    return render_template("index.html", result=result, active_tab="image")


# ── Audio Watermarking ──────────────────────────────────────────────────────

@app.route("/process-audio", methods=["POST"])
def process_audio():
    audio_file = request.files.get("audio")
    image_file = request.files.get("watermark_image")

    if not audio_file or not image_file:
        return render_template(
            "index.html",
            error="Please upload both a WAV audio file and a watermark image.",
            active_tab="audio",
        )

    audio_path = os.path.join(UPLOAD_DIR, audio_file.filename)
    image_path = os.path.join(UPLOAD_DIR, image_file.filename)
    audio_file.save(audio_path)
    image_file.save(image_path)

    try:
        audio_results = process_audio_watermark(audio_path, image_path, OUTPUT_DIR)
    except Exception as exc:
        return render_template(
            "index.html",
            error=f"Audio processing failed: {exc}",
            active_tab="audio",
        )

    # Normalise paths for URL usage
    for r in audio_results:
        r["audio_path"] = r["audio_path"].replace("\\", "/")
        r["extracted_image"] = r["extracted_image"].replace("\\", "/")

    return render_template("index.html", audio_results=audio_results, active_tab="audio")


# ── Static file download helper ─────────────────────────────────────────────

@app.route("/download/<path:filename>")
def download(filename):
    return send_from_directory("static", filename, as_attachment=True)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
