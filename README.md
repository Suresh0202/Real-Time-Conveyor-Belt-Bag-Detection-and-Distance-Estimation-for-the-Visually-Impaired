# Smart Vision Assistive System using YOLOv8

### 🔍 Bag Detection with QR & TTS for Visually Impaired

This project detects travel bags (suitcases, backpacks, handbags) in video using **YOLOv8**, overlays a **QR code**, and uses **text-to-speech (TTS)** to speak distance and bag info — useful for visually impaired users.

---

## 🧠 Features

- Real-time object detection using `YOLOv8n`
- QR code overlay on detected bags using `qrcode + OpenCV`
- Distance estimation using bounding box size
- Voice alerts using `pyttsx3`
- Detection caching for smooth box display

---

## 📁 Files Included

- `main.py` - Core Python code (fully commented)
- `final_presentation_suresh.pdf` - 4-slide summary
- `README.md` - Project overview (this file)

*Videos and large models like `.pt` should be linked from Google Drive.*

---

## ▶️ Demo Video

📽️ Watch the demo here: [Google Drive Link](https://drive.google.com/your-demo-link)

---

## 💻 How to Run

```bash
pip install opencv-python pyttsx3 qrcode numpy ultralytics
python main.py
