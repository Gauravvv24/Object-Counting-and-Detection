
# 🎥 Object Counting and Detection – YOLOv11x vs YOLOv9e vs YOLOv8x

This Streamlit app enables side-by-side comparison and performance analytics of different YOLO models (YOLOv11x, YOLOv9e, YOLOv8x) on a single uploaded video. It visualizes object detection, tracks region crossings, and provides frame-by-frame insights and downloadable analytics.

## 🚀 Features

- 🔍 **Model Comparison**: Compare object detection outputs from YOLOv11x, YOLOv9e, and YOLOv8x
- 🎯 **Region Type Selection**: Choose between Line or Rectangle region for crossing detection
- 📊 **Real-Time Analytics**:
  - Frame-by-frame crossing counts
  - Total object crossings per model
  - Average inference time per frame
- 🎬 **Side-by-Side Video Outputs**
- 🔁 **Frame Preview Slider**
- 💾 **Export Metrics**: Download `.csv` files for summary and frame-wise counts

## 🛠 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Gauravvv24/Object-Counting-and-Detection.git
   cd Object-Counting-and-Detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Place your YOLO `.pt` model files in the root directory or provide full paths in the app.

## ▶️ Run the App

```bash
streamlit run app.py
```

## 📁 Project Structure

```
.
├── app.py                 # Main Streamlit app
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## 🧠 Supported YOLO Models

- YOLOv11x
- YOLOv9e
- YOLOv8x  
*(Models must be trained/downloaded via [Ultralytics](https://github.com/ultralytics/ultralytics))*

## 📌 Usage Notes

- Supported formats: `.mp4`, `.avi`, `.mov`
- Region Types:
  - **Line** – Detects single-line crossings (ideal for narrow lanes or gates)
  - **Rectangle** – Detects entries and exits within a defined box region

## 📊 Output

- **Videos** with real-time visualized detections
- **CSV Reports**:
  - `summary.csv` – Total crossings + inference time per model
  - `counts.csv` – Frame-wise cumulative crossing counts

## 🧩 Future Scope

- Real-time webcam input
- Integration with more object detection frameworks
- Object-specific analytics (e.g., class-wise detection)

## 📬 Contact

Built by [Gaurav Singh Khati](mailto:khatigaurav8@gmail.com)  
GitHub: [github.com/Gauravvv24](https://github.com/Gauravvv24)  
Project: [Object Counting and Detection](https://github.com/Gauravvv24/Object-Counting-and-Detection)
