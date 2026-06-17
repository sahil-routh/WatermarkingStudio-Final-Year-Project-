"""
Audio LSB Watermarking (WAV cover + Image watermark)
Extracted and refactored from new.py into a proper importable module.

Automatically runs 1-bit, 4-bit and 7-bit LSB embedding and returns
results for all three in a list.
"""

import numpy as np
from scipy.io import wavfile
from PIL import Image
import os
import math


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _lsb_embed(signal: np.ndarray, wm_bits: str, b: int) -> np.ndarray:
    """Replace the b LSBs of each audio sample with chunks of wm_bits."""
    mask = (1 << b) - 1
    num_samples_needed = (len(wm_bits) + b - 1) // b
    if num_samples_needed > len(signal):
        raise ValueError(
            f"Audio too short: need {num_samples_needed} samples, have {len(signal)}"
        )
    signal = signal.copy()
    for i in range(num_samples_needed):
        chunk = wm_bits[i * b : (i + 1) * b]
        if len(chunk) < b:
            chunk = chunk.ljust(b, "0")
        signal[i] = (signal[i] & ~mask) | int(chunk, 2)
    return signal


def _lsb_extract(signal: np.ndarray, b: int, num_bits: int) -> str:
    """Read num_bits worth of LSB data from signal at b bits per sample."""
    mask = (1 << b) - 1
    num_samples_needed = (num_bits + b - 1) // b
    bits = [f"{signal[i] & mask:0{b}b}" for i in range(num_samples_needed)]
    return "".join(bits)[:num_bits]


def _mse_psnr(original: np.ndarray, modified: np.ndarray, max_val: int):
    """Return (MSE, PSNR-dB) rounded to 4 / 2 decimal places."""
    diff = original.astype(np.float64) - modified.astype(np.float64)
    mse_val = np.mean(diff ** 2)
    psnr_val = float("inf") if mse_val == 0 else 10 * np.log10((max_val ** 2) / mse_val)
    return round(float(mse_val), 4), round(float(psnr_val), 2)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_audio_watermark(audio_path: str, image_path: str, output_dir: str) -> list:
    """
    Embed an image into a WAV audio file using 1-bit, 4-bit and 7-bit LSB.

    Parameters
    ----------
    audio_path  : path to the cover WAV file
    image_path  : path to the watermark image (any PIL-readable format)
    output_dir  : directory where output files will be written

    Returns
    -------
    list of dicts, one per bit-depth, each containing:
        bits            int   – LSB depth used
        audio_path      str   – path to watermarked WAV file
        audio_filename  str   – basename of the watermarked WAV
        extracted_image str   – path to the reconstructed image PNG
        image_size      str   – "WxH" of the embedded image
        mse             float – audio MSE (original vs watermarked)
        psnr            float – audio PSNR in dB
        bits_embedded   int   – total bits hidden in the audio
        capacity        int   – total bit capacity for this b
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- Load audio ---
    fs, audio_orig = wavfile.read(audio_path)
    if audio_orig.ndim == 2:
        audio_orig = audio_orig[:, 0]          # take left channel only
    audio_orig = audio_orig.astype(np.int16)
    max_audio = 32767
    n_samples = len(audio_orig)

    # --- Load image ---
    img_original = Image.open(image_path).convert("RGB")

    results = []

    for b in (1, 4, 7):
        max_bits_capacity = n_samples * b
        max_bytes_capacity = max_bits_capacity // 8

        if max_bytes_capacity <= 0:
            continue

        # Scale image down to fit capacity while preserving aspect ratio
        orig_w, orig_h = img_original.size
        orig_pixels = orig_w * orig_h * 3         # RGB channels

        scale = math.sqrt(max_bytes_capacity / orig_pixels)
        if scale >= 1.0:
            img_resized = img_original.copy()
        else:
            new_w = max(1, int(orig_w * scale))
            new_h = max(1, int(orig_h * scale))
            img_resized = img_original.resize((new_w, new_h), Image.LANCZOS)

        img_wm_np = np.array(img_resized)
        h, w, c = img_wm_np.shape

        img_flat = img_wm_np.flatten().astype(np.uint8)
        wm_bits = "".join(f"{p:08b}" for p in img_flat)
        num_wm_bits = len(wm_bits)

        # Clamp to actual capacity
        if num_wm_bits > max_bits_capacity:
            wm_bits = wm_bits[:max_bits_capacity]
            num_wm_bits = len(wm_bits)
            usable_bytes = num_wm_bits // 8
            img_flat = img_flat[:usable_bytes]
            extra = (h * w * 3) - usable_bytes
            if extra > 0:
                img_flat = np.pad(img_flat, (0, extra), "constant")

        # --- Embed ---
        try:
            audio_embedded = _lsb_embed(audio_orig, wm_bits, b)
        except ValueError:
            continue

        mse_val, psnr_val = _mse_psnr(audio_orig, audio_embedded, max_audio)

        audio_out_name = f"watermarked_audio_{b}bit.wav"
        audio_out_path = os.path.join(output_dir, audio_out_name)
        wavfile.write(audio_out_path, fs, audio_embedded)

        # --- Extract & reconstruct image ---
        ext_bits = _lsb_extract(audio_embedded, b, num_wm_bits)
        ext_vals = np.array(
            [int(ext_bits[i : i + 8], 2) for i in range(0, num_wm_bits, 8)],
            dtype=np.uint8,
        )

        needed = h * w * 3
        if ext_vals.size < needed:
            ext_vals = np.pad(ext_vals, (0, needed - ext_vals.size), "constant")
        else:
            ext_vals = ext_vals[:needed]

        ext_img_name = f"extracted_from_audio_{b}bit.png"
        ext_img_path = os.path.join(output_dir, ext_img_name)
        Image.fromarray(ext_vals.reshape((h, w, 3))).save(ext_img_path)

        results.append(
            {
                "bits": b,
                "audio_path": audio_out_path,
                "audio_filename": audio_out_name,
                "extracted_image": ext_img_path,
                "image_size": f"{w}×{h}",
                "mse": mse_val,
                "psnr": psnr_val,
                "bits_embedded": num_wm_bits,
                "capacity": max_bits_capacity,
            }
        )

    return results
