# YOLO-Based Vehicle Detection

A computer vision project that uses the **YOLO (You Only Look Once) object detection algorithm** to detect and localize vehicles in images and video streams. The project applies deep learning and OpenCV to identify vehicles, draw bounding boxes, and display detection confidence scores.

## 🚀 Features

- **Vehicle Detection:** Identifies vehicles using a trained YOLO model.
- **Video Processing:** Supports vehicle detection on video footage.
- **Bounding Box Visualization:** Highlights detected objects with bounding boxes and confidence scores.
- **ONNX Model Inference:** Uses OpenCV's DNN module to load and run an ONNX model.
- **Dataset Configuration:** Uses a YAML configuration file to define dataset class labels.
- **Model Experimentation:** Includes Jupyter notebooks for detection experiments, dataset preparation, and testing.

## 🛠️ Tech Stack

- **Language:** Python
- **Computer Vision:** OpenCV
- **Object Detection:** YOLO
- **Model Format:** ONNX
- **Data Processing:** NumPy, PyYAML
- **Development Environment:** Jupyter Notebook / Google Colab

## 📂 Project Structure

```text
yolo-cv-project/
│
├── car_detection.py
├── car_detection.ipynb
├── final_predictions.ipynb
├── extract_object_info_from_xml.ipynb
├── test.py
├── test.ipynb
├── data.yaml
├── requirements.txt
├── car_parking.mp4
├── video.mp4
└── README.md
```

**Key files:**

- `car_detection.py` — Implements vehicle detection using OpenCV and a YOLO model in ONNX format.
- `car_detection.ipynb` — Notebook for developing and experimenting with vehicle detection.
- `final_predictions.ipynb` — Notebook for running and examining predictions.
- `extract_object_info_from_xml.ipynb` — Supports extracting object information from XML annotations.
- `test.py` and `test.ipynb` — Used for testing detection functionality.
- `data.yaml` — Stores dataset class-label configuration.
- `requirements.txt` — Lists the Python dependencies.
- `car_parking.mp4` and `video.mp4` — Sample video files for experimentation.

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/LordCenk/yolo-cv-project.git
cd yolo-cv-project
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows**
```bash
venv\Scripts\activate
```

**Linux / macOS**
```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## ▶️ Usage

### Run vehicle detection

```bash
python car_detection.py
```

Ensure the trained ONNX model and dataset configuration are available at the paths expected by the script. The detector references `Model/weights/best.onnx` and `data.yaml`.

If necessary, update the model, input-video, and output paths in the code to match your local setup.

### Run using Jupyter Notebook

Launch Jupyter:

```bash
jupyter notebook
```

Open `car_detection.ipynb` or `final_predictions.ipynb` and execute the cells in sequence. If working in Google Colab, upload or mount the required model, dataset, and video files.

## 🔍 How It Works

1. **Input:** Loads an image or video frame.
2. **Preprocessing:** Resizes and normalizes the image for model inference.
3. **Detection:** Passes the processed image through the trained YOLO model.
4. **Post-processing:** Filters predictions using confidence thresholds and Non-Maximum Suppression (NMS).
5. **Visualization:** Draws bounding boxes and displays class labels with confidence scores.

## 🌍 Potential Applications

- Intelligent traffic monitoring
- Vehicle detection in parking areas
- Smart transportation systems
- Traffic surveillance and analysis
- Computer vision research and experimentation

## 🔮 Future Improvements

- Add vehicle counting and traffic-density estimation.
- Track vehicles across video frames.
- Evaluate model performance using precision, recall, and mAP.
- Optimize inference speed for real-time applications.
- Build a web interface for uploading videos and viewing predictions.

## 📌 Limitations

Detection accuracy and inference speed depend on the trained model, input-video quality, lighting conditions, and available hardware. The trained model and its corresponding dataset must be provided separately if they are not included in the repository.

## 👨‍💻 Author

**Shashank Agrawal**

GitHub: [@LordCenk](https://github.com/LordCenk)

