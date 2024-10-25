import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import torch
import math

class ObjectCounter:
    def __init__(self, capture_source, line_position=0.5):
        """
        Initialize Object Counter
        Args:
            capture_source: Video file path or camera index
            line_position: Position of counting line (0 to 1)
        """
        self.model = YOLO('yolov8n.pt')
        self.capture = cv2.VideoCapture(capture_source)
        self.frame_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.capture.get(cv2.CAP_PROP_FPS))
        
        # Define counting line
        self.line_position = int(self.frame_height * line_position)
        self.line_points = [(0, self.line_position), (self.frame_width, self.line_position)]
        
        # Initialize counters
        self.counter = defaultdict(int)
        self.tracked_objects = {}
        
        # Set up video writer
        self.output_path = 'output_counted.mp4'
        self.writer = cv2.VideoWriter(
            self.output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            self.fps,
            (self.frame_width, self.frame_height)
        )
        
        # Colors for visualization
        self.colors = np.random.randint(0, 255, size=(90, 3)).tolist()

    def process_frame(self, frame, results):
        """
        Process detection results and draw visualizations
        """
        # Draw counting line
        cv2.line(frame, self.line_points[0], self.line_points[1], (0, 255, 0), 2)
        
        if not results or len(results) == 0:
            return frame
            
        boxes = results[0].boxes
        
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Get object center
            center_y = (y1 + y2) // 2
            
            # Get class details
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = self.model.names[class_id]
            
            # Get tracking ID if available
            track_id = int(box.id[0]) if box.id is not None else -1
            
            # Count object if it crosses the line
            if track_id != -1:
                if track_id in self.tracked_objects:
                    prev_center_y = self.tracked_objects[track_id]
                    # Check if object crossed the line
                    if (prev_center_y < self.line_position and center_y >= self.line_position) or \
                       (prev_center_y > self.line_position and center_y <= self.line_position):
                        self.counter[class_name] += 1
                
                self.tracked_objects[track_id] = center_y
            
            # Draw bounding box
            color = self.colors[class_id]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f'{class_name} {conf:.2f}'
            t_size = cv2.getTextSize(label, 0, 0.7, 2)[0]
            cv2.rectangle(frame, (x1, y1 - t_size[1] - 3), (x1 + t_size[0], y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 2), 0, 0.7, (255, 255, 255), 2)
        
        # Draw counter
        y_offset = 30
        for class_name, count in self.counter.items():
            text = f'{class_name}: {count}'
            cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            y_offset += 30
            
        return frame

    def run(self):
        """
        Main processing loop
        """
        try:
            while self.capture.isOpened():
                success, frame = self.capture.read()
                if not success:
                    break
                
                # Run detection with tracking
                results = self.model.track(frame, persist=True, verbose=False)
                
                # Process frame and draw visualizations
                processed_frame = self.process_frame(frame, results)
                
                # Display frame
                cv2.imshow('Object Counting', processed_frame)
                
                # Write frame to output video
                self.writer.write(processed_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            # Cleanup
            self.capture.release()
            self.writer.release()
            cv2.destroyAllWindows()
            
    def get_counts(self):
        """
        Return the current counts
        """
        return dict(self.counter)

def main():
    # Initialize and run counter
    # Use 0 for webcam or provide video file path
    counter = ObjectCounter(capture_source="video.mp4")
    counter.run()
    
    # Print final counts
    print("\nFinal Counts:")
    for class_name, count in counter.get_counts().items():
        print(f"{class_name}: {count}")

if __name__ == "__main__":
    main()