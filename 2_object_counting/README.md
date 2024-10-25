# Real-time Object Counter with YOLOv8

A sophisticated real-time object detection and counting system using YOLOv8. This system features line-crossing detection to count objects and provides detailed visualizations and statistics.

## Features

### Core Functionality
- Real-time object detection and tracking
- Line-crossing detection for accurate counting
- Multi-class object tracking
- Directional counting (both upward and downward crossing)
- Frame-by-frame processing with real-time visualization
- Video output generation with annotations

### Counter Features
- Per-class object counting
- Bi-directional counting capability
- Persistent tracking across frames
- Real-time count display
- Customizable counting line position

### Visualization
- Bounding boxes with class labels
- Confidence scores
- Color-coded object categories
- Counting line display
- Real-time counter overlay

## Requirements

### Hardware
- CPU: Intel i5 or equivalent (minimum)
- RAM: 8GB minimum (16GB recommended)
- GPU: NVIDIA GPU with CUDA support (recommended)
- Webcam/Video input device (for live capture)

### Software Dependencies
```bash
Python >= 3.8
OpenCV >= 4.5
PyTorch >= 1.7
Ultralytics >= 8.0
NumPy >= 1.19
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Elite-AI-Club/AI-Driven-Innovation-Electronics.git
cd object-counter
```

2. Create and activate virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/MacOS
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage
```python
from object_counter import ObjectCounter

# Initialize counter with video source
counter = ObjectCounter(capture_source="path/to/video.mp4")

# Run counting
counter.run()

# Get final counts
counts = counter.get_counts()
```

### Configuration Options

1. Line Position:
```python
# Set counting line at 70% of frame height
counter = ObjectCounter(capture_source="video.mp4", line_position=0.7)
```

2. Video Output:
```python
# Custom output path
self.output_path = 'my_counted_video.mp4'
```

3. Display Settings:
```python
# Customize visualization
cv2.putText(frame, text, (x, y), font, scale, color, thickness)
```

## Class Documentation

### ObjectCounter Class

#### Initialization
```python
def __init__(self, capture_source, line_position=0.5):
    """
    Initialize Object Counter
    Args:
        capture_source: Video file path or camera index
        line_position: Position of counting line (0 to 1)
    """
```

#### Key Methods

1. `process_frame(frame, results)`:
   - Processes detection results
   - Updates object counts
   - Draws visualizations
   - Returns annotated frame

2. `run()`:
   - Main processing loop
   - Handles video I/O
   - Manages real-time display

3. `get_counts()`:
   - Returns current count statistics
   - Provides per-class counts

## Project Structure
```
object-counter/
│
├── object_counter.py    # Main implementation
├── requirements.txt     # Project dependencies
├── README.md           # Project documentation
├── models/             # YOLO model weights
│   └── yolov8n.pt
├── input/              # Input videos
└── output/             # Output videos and statistics
```

## Implementation Details

### Object Tracking
```python
# Track objects across frames
results = self.model.track(frame, persist=True, verbose=False)
```

### Line Crossing Detection
```python
if (prev_center_y < self.line_position and center_y >= self.line_position) or \
   (prev_center_y > self.line_position and center_y <= self.line_position):
    self.counter[class_name] += 1
```

### Visualization
```python
# Draw bounding box
cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

# Draw label
cv2.putText(frame, label, (x1, y1 - 2), 0, 0.7, (255, 255, 255), 2)
```

## Performance Optimization

1. Frame Processing
   - Efficient line crossing detection
   - Optimized tracking algorithm
   - Real-time visualization

2. Memory Management
   - Efficient object tracking storage
   - Optimized video writing
   - Memory cleanup on exit

3. GPU Utilization
   - CUDA acceleration for detection
   - Batch processing capability
   - Efficient memory usage

## Troubleshooting

### Common Issues

1. Video Input
```python
# Check if video is opened successfully
assert self.capture.isOpened(), "Error: Could not open video."
```

2. Performance
```python
# Reduce frame size for better performance
frame = cv2.resize(frame, (640, 480))
```

3. Memory
```python
# Clean up resources
self.capture.release()
self.writer.release()
cv2.destroyAllWindows()
```

## Future Improvements

- [ ] Multiple counting lines
- [ ] Direction-based counting
- [ ] Speed estimation
- [ ] Database integration
- [ ] Web interface
- [ ] Multi-camera support
- [ ] Custom model training
- [ ] Real-time analytics

## Contributing

1. Fork the repository
2. Create feature branch 
3. Commit changes 
4. Push to branch 
5. Open Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- YOLOv8 by Ultralytics
- OpenCV community
- PyTorch team

## Contact
- https://github.com/jidhu-mohan
Project Link: https://github.com/Elite-AI-Club/AI-Driven-Innovation-Electronics
