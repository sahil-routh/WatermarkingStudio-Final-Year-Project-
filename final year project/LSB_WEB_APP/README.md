# LSB_WEB_APP
A web-based implementation of Image Watermarking using the Least Significant Bit (LSB) algorithm with quality analysis using MSE and PSNR. Built using Python, Flask, NumPy, and PIL.

# Image Watermarking using LSB Algorithm (Web-Based)

This project implements **Image Watermarking using the Least Significant Bit (LSB) algorithm** with a **web-based front end**.  
The system allows users to embed a watermark image into a cover image, extract the watermark, and evaluate image quality using **MSE (Mean Square Error)** and **PSNR (Peak Signal-to-Noise Ratio)**.

The application is built using **Python (Flask)** for the backend and **HTML/CSS** for the frontend, making it interactive and easy to demonstrate for academic and practical use.

---

## 🔍 Features

- Upload **cover image** and **watermark image**
- Embed watermark using **LSB (4-bit) technique**
- Extract watermark from the watermarked image
- Compute **MSE and PSNR** for image quality analysis
- Clean and responsive **web interface**
- Modular and easy-to-understand code structure

---

## 🧠 Concept Used

### Least Significant Bit (LSB) Watermarking
- The **least significant bits** of the cover image pixels are replaced with the **most significant bits** of the watermark image.
- This ensures:
  - Minimal visual distortion
  - High imperceptibility
  - Simple and fast computation

### Image Quality Metrics
- **MSE (Mean Square Error)**: Measures average squared difference between images  
- **PSNR (Peak Signal-to-Noise Ratio)**: Measures image quality (higher is better)

---

## 🛠️ Technologies Used

- **Python 3.11**
- **Flask** – Web framework
- **NumPy** – Numerical computations
- **Pillow (PIL)** – Image processing
- **HTML & CSS** – Frontend UI

---

## 📁 Project Structure

