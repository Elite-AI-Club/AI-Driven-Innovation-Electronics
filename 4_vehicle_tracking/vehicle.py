import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import time

class VehicleTracker:
    def __init__(self, video_path, model_path='yolov8n-seg.pt'):
        """
        Initialize Vehicle Tracker
        Args:
            video_path: Path to input video file
            model_path: Path to YOLO segmentation model
        """
        self.video_path = video_path
        self.model = YOLO(model_path)
        self.track_history = defaultdict(lambda: [])
        self.vehicle_counts = defaultdict(int)
        self.tracked_vehicles = set()
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(video_path)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        # Initialize video writer
        self.output_path = 'vehicle_tracking_output.avi'
        self.writer = cv2.VideoWriter(
            self.output_path,
            cv2.VideoWriter_fourcc(*'MJPG'),
            self.fps,
            (self.frame_width, self.frame_height)
        )
        
        # Define vehicle classes (customize based on your needs)
        self.vehicle_classes = {'car', 'truck', 'bus', 'motorcycle'}

    def process_frame(self, frame, results):
        """
        Process frame and draw annotations
        """
        annotator = Annotator(frame, line_width=2)
        
        if results[0].boxes.id is not None and results[0].masks is not None:
            masks = results[0].masks.xy
            boxes = results[0].boxes
            track_ids = boxes.id.int().cpu().tolist()
            classes = boxes.cls.int().cpu().tolist()
            
            # Process each detected object
            for mask, track_id, cls in zip(masks, track_ids, classes):
                class_name = self.model.names[cls]
                
                # Only process vehicles
                if class_name in self.vehicle_classes:
                    # Update vehicle counts
                    if track_id not in self.tracked_vehicles:
                        self.tracked_vehicles.add(track_id)
                        self.vehicle_counts[class_name] += 1
                    
                    # Draw segmentation mask and label
                    mask_color = colors(track_id, True)
                    label = f"{class_name}-{track_id}"
                    annotator.seg_bbox(
                        mask=mask,
                        mask_color=mask_color,
                        label=label
                    )
                    
                    # Update tracking history
                    center = np.mean(mask, axis=0)
                    self.track_history[track_id].append(center)
                    
                    # Draw tracking trail
                    if len(self.track_history[track_id]) > 1:
                        trail = np.array(self.track_history[track_id]).astype(np.int32)
                        cv2.polylines(frame, [trail], False, mask_color, 2)
        
        # Draw vehicle counts
        y_offset = 30
        for vehicle_type, count in self.vehicle_counts.items():
            text = f"{vehicle_type}: {count}"
            cv2.putText(frame, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            y_offset += 30
        
        return frame

    def run(self):
        """
        Main processing loop
        """
        frame_count = 0
        start_time = time.time()
        
        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Process every other frame for better performance
                frame_count += 1
                if frame_count % 2 != 0:
                    continue
                
                # Perform tracking
                results = self.model.track(
                    frame,
                    persist=True,
                    show=False,
                    classes=[2, 3, 5, 7]  # Filter for vehicle classes
                )
                
                # Process and annotate frame
                processed_frame = self.process_frame(frame, results)
                
                # Calculate and display FPS
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time
                cv2.putText(processed_frame, f'FPS: {int(fps)}', 
                           (10, self.frame_height - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Display frame
                cv2.imshow('Vehicle Tracking', processed_frame)
                
                # Write frame to output video
                self.writer.write(processed_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            # Cleanup
            self.cap.release()
            self.writer.release()
            cv2.destroyAllWindows()
            
        return self.vehicle_counts

    def get_statistics(self):
        """
        Return tracking statistics
        """
        return {
            'total_vehicles': sum(self.vehicle_counts.values()),
            'vehicle_counts': dict(self.vehicle_counts),
            'unique_vehicles': len(self.tracked_vehicles)
        }

def main():
    # Initialize tracker
    tracker = VehicleTracker(
        video_path="Test Video.mp4",
        model_path="yolov8m-seg.pt"
    )
    
    # Run tracking
    tracker.run()
    
    # Print statistics
    stats = tracker.get_statistics()
    print("\nTracking Statistics:")
    print(f"Total Vehicles: {stats['total_vehicles']}")
    print("\nVehicle Counts by Type:")
    for vehicle_type, count in stats['vehicle_counts'].items():
        print(f"{vehicle_type}: {count}")
    print(f"\nUnique Vehicles Tracked: {stats['unique_vehicles']}")

if __name__ == "__main__":
    main()