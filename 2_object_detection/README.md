# Real-time Object Detection using YOLO

This project implements real-time object detection using YOLOv8 and OpenCV. It captures video from your webcam and performs object detection on each frame, displaying the results with bounding boxes and labels.

## Features
- Real-time object detection using webcam
- FPS (Frames Per Second) counter
- Color-coded bounding boxes for different object classes
- Confidence score display for each detection
- Support for 80+ object classes from COCO dataset
- Easy-to-use interface with simple keyboard controls

## Prerequisites
- Python 3.8 or higher
- Webcam (built-in or external)
- GPU (optional but recommended for better performance)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Elite-AI-Club/AI-Driven-Innovation-Electronics.git
cd realtime-object-detection
```

2. Create a virtual environment (recommended):
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/MacOS
python3 -m venv venv
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Dependencies
Create a `requirements.txt` file with the following contents:
```
ultralytics>=8.0.0
opencv-python>=4.8.0
numpy>=1.22.0
```

## Usage

1. Run the main script:
```bash
python vehicle.py
```

2. Controls:
- Press 'q' to quit the application
- The FPS counter is displayed in the top-left corner
- Detection boxes show object class and confidence percentage

## Configuration

### Changing Resolution
Modify these lines in the code to change the webcam resolution:
```python
cap.set(3, 640)  # width
cap.set(4, 480)  # height
```

### Changing YOLO Model
The project uses YOLOv8n (nano) by default. To use a different model, modify:
```python
model = YOLO('yolov8n.pt')
```

Available models:
- `yolov8n.pt` (nano) - fastest
- `yolov8s.pt` (small) - balanced
- `yolov8m.pt` (medium) - more accurate
- `yolov8l.pt` (large) - even more accurate
- `yolov8x.pt` (extra large) - most accurate

## Project Structure
```
realtime-object-detection/
│
├── detect.py           # Main detection script
├── requirements.txt    # Project dependencies
├── README.md          # Project documentation
└── .gitignore         # Git ignore file
```

## Detectable Objects
The system can detect 80+ different object classes including:
- People
- Vehicles (cars, trucks, bicycles, etc.)
- Animals (dogs, cats, birds, etc.)
- Electronics (laptops, phones, etc.)
- Furniture
- Common household items

## Performance Tips
1. Use a GPU for better performance
2. Reduce resolution for higher FPS
3. Use smaller YOLO models (nano/small) for faster detection
4. Ensure good lighting conditions
5. Keep webcam lens clean

## Troubleshooting

### Common Issues and Solutions

1. Webcam not detected:
```python
# Try changing the camera index
cap = cv2.VideoCapture(1)  # or try 2, 3, etc.
```

2. Low FPS:
- Reduce resolution
- Use YOLOv8n model
- Check CPU/GPU usage
- Close unnecessary applications

3. ModuleNotFoundError:
- Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

4. CUDA/GPU Issues:
- Install appropriate CUDA toolkit
- Update GPU drivers
- Verify PyTorch GPU support:
```python
import torch
print(torch.cuda.is_available())
```

## Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- YOLOv8 by Ultralytics
- OpenCV community
- COCO dataset creators

## Contact

- GitHub: https://github.com/jidhu-mohan
- Project Link: https://github.com/Elite-AI-Club/AI-Driven-Innovation-Electronics

## Future Improvements
- [ ] Add object tracking
- [ ] Implement video recording
- [ ] Add custom object detection
- [ ] Improve UI with controls
- [ ] Add multiple camera support
