# Vehicle Tracking System with YOLOv8

A sophisticated vehicle tracking system that uses YOLOv8 with instance segmentation to detect, track, and count vehicles in video footage. The system provides real-time visualization of vehicle movements and maintains detailed statistics of vehicle counts.

## Features

### Core Functionality
- Real-time vehicle detection and tracking
- Instance segmentation for precise vehicle boundaries
- Multi-class vehicle tracking (cars, trucks, buses, motorcycles)
- Unique vehicle identification and counting
- Movement trail visualization
- Real-time FPS monitoring
- Video output generation with annotations

### Vehicle Statistics
- Total vehicle count
- Per-class vehicle counting
- Unique vehicle tracking
- Real-time count display
- Historical movement tracking

### Visualization
- Color-coded instance segmentation masks
- Vehicle tracking trails
- Class labels with tracking IDs
- Real-time statistics overlay
- Performance metrics display

## Requirements

### Hardware
- CPU (minimum): Intel i5 or equivalent
- RAM: 8GB minimum (16GB recommended)
- GPU: NVIDIA GPU with CUDA support (recommended)
- Webcam/Video input device (for live capture)

### Software
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
cd vehicle_tracking
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
from vehicle_tracker import VehicleTracker

# Initialize tracker
tracker = VehicleTracker(
    video_path="video.mp4",
    model_path="yolov8m-seg.pt"
)

# Run tracking
tracker.run()
```

### Configuration Options
```python
# Custom vehicle classes
tracker.vehicle_classes = {'car', 'truck', 'bus'}

# Adjust frame processing rate
frame_count % 3 != 0  # Process every third frame

# Change output video format
self.writer = cv2.VideoWriter(
    'output.mp4',
    cv2.VideoWriter_fourcc(*'MP4V'),
    fps,
    (width, height)
)
```


## Class Documentation

### VehicleTracker Class

#### Initialization
```python
def __init__(self, video_path, model_path='yolov8n-seg.pt'):
    """
    Initialize Vehicle Tracker
    Args:
        video_path: Path to input video file
        model_path: Path to YOLO segmentation model
    """
```

#### Key Methods
1. `process_frame(frame, results)`:
   - Processes each video frame
   - Draws annotations and tracking information
   - Updates vehicle counts

2. `run()`:
   - Main processing loop
   - Handles video input/output
   - Manages real-time display

3. `get_statistics()`:
   - Returns tracking statistics
   - Provides vehicle counts and metrics

## Performance Optimization

1. Frame Processing
   - Skip frames for better performance
   - Optimize for specific vehicle classes
   - Balance between accuracy and speed

2. Memory Management
   - Limited tracking history
   - Efficient mask processing
   - Optimized video writing

3. GPU Acceleration
   - CUDA support for YOLO model
   - Batch processing capability
   - Hardware acceleration for video processing

## Troubleshooting

### Common Issues

1. Memory Usage
```python
# Limit tracking history
if len(self.track_history[track_id]) > 30:
    self.track_history[track_id].pop(0)
```

2. Performance Issues
```python
# Reduce processing load
frame = cv2.resize(frame, (640, 480))
```

3. GPU Memory
```python
# Free GPU memory
torch.cuda.empty_cache()
```

## Contributing

1. Fork the repository
2. Create feature branch 
3. Commit changes 
4. Push to branch 
5. Open Pull Request

## Future Improvements

- [ ] Speed estimation
- [ ] Direction-based counting
- [ ] Multi-camera support
- [ ] Database integration
- [ ] Web interface
- [ ] Real-time analytics
- [ ] Custom model training
- [ ] API integration

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- YOLOv8 by Ultralytics
- OpenCV community
- PyTorch team

