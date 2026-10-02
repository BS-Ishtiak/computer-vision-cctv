# CCTV AI Project - Computer Vision Learning

A real-time CCTV system for face recognition and person detection using InsightFace, FAISS, and YOLO.

## 🎯 Features

✅ **Completed:**
- InsightFace face detection
- Face embeddings extraction
- FAISS face database
- Real-time webcam face recognition
- YOLO person detection

🔜 **Coming Soon:**
- ByteTrack/BoxMOT person tracking
- Combine tracking + face recognition
- Multi-person event logging
- Event storage and replay

---

## 📋 Project Structure

```
cctv-ai-project/
├── venv/                      # Virtual environment (git ignored)
├── faces/                      # Sample face images for database
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore file
├── bus.jpg                     # Test image
│
├── create_face_db.py           # Build face database from images
├── face_detection.py           # Test face detection with InsightFace
├── face_embedding.py           # Extract face embeddings
├── recognize_webcam.py         # Real-time face recognition from webcam
├── person_detection.py         # YOLO person detection
├── test_yolo.py                # Test YOLO model
├── track.py                    # Tracking tests
└── webcam.py                   # Webcam utilities
```

---

## 🚀 Quick Start

### **1. Setup Virtual Environment**
```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3. Prepare Face Database**
```bash
# Put face images in faces/ folder
# Folder structure: faces/person_name/image.jpg
# Example: faces/john/photo1.jpg, faces/john/photo2.jpg

python create_face_db.py
```

### **4. Run Face Recognition**
```bash
python recognize_webcam.py
```
Press `q` to quit.

### **5. Run Person Detection**
```bash
python person_detection.py
```
Press `q` to quit.

---

## 📖 File Explanations

| File | Purpose |
|------|---------|
| `create_face_db.py` | Creates FAISS index from face images in `faces/` folder |
| `face_detection.py` | Tests InsightFace detection on a single image |
| `face_embedding.py` | Extracts and displays face embeddings |
| `recognize_webcam.py` | Real-time webcam face recognition with FAISS |
| `person_detection.py` | **NEW** - YOLO-based person detection in real-time |

---

## 🔧 How It Works

### **Face Recognition Pipeline**
1. InsightFace detects faces in frame
2. Extracts 512-dim face embeddings
3. Searches FAISS database for similar faces
4. Returns match with confidence score

### **Person Detection Pipeline**
1. YOLO scans frame for objects
2. Filters for class 0 (person)
3. Draws bounding boxes around detected people
4. Shows confidence scores

---

## 💻 Hardware Requirements

- **GPU (Recommended):** NVIDIA GPU for faster inference
- **CPU (Fallback):** Works but slower
- **RAM:** 4GB minimum
- **Webcam:** Standard USB or built-in camera

---

## 📦 Dependencies

- **OpenCV** - Image/video processing
- **InsightFace** - Face detection & embeddings
- **FAISS** - Face database & similarity search
- **YOLO (Ultralytics)** - Person detection
- **NumPy** - Numerical computing

---

## 🎓 Learning Resources

- [InsightFace GitHub](https://github.com/deepinsight/insightface)
- [FAISS Documentation](https://faiss.ai/)
- [YOLOv8 Guide](https://docs.ultralytics.com/)
- [OpenCV Tutorials](https://docs.opencv.org/)

---

## 👨‍💻 Author

**BS-Ishtiak**  
Computer Vision Learning Project

---

## 📝 License

MIT License

---

## 🤝 Next Steps

1. ✅ Face recognition working
2. ✅ Person detection working
3. ⏳ Add tracking (ByteTrack)
4. ⏳ Combine detection + tracking + recognition
5. ⏳ Add event logging database
