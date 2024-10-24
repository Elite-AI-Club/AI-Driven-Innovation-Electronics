import cv2
from ultralytics import YOLO
import numpy as np
import time

def main():
    # Load the YOLOv8 model
    model = YOLO('yolov8n.pt')

    # Open the webcam (0 is usually the default webcam)
    cap = cv2.VideoCapture(0)
    
    # Set webcam resolution
    cap.set(3, 640)  # width
    cap.set(4, 480)  # height

    # Initialize variables for FPS calculation
    prev_time = 0
    
    # Generate random colors for different classes
    colors = np.random.uniform(0, 255, size=(80, 3))

    while cap.isOpened():
        # Read a frame from the webcam
        success, frame = cap.read()

        if success:
            # Run YOLOv8 inference on the frame
            results = model(frame)

            # Calculate FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time

            # Draw FPS on frame
            cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

            # Visualize the results on the frame
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Get confidence
                    conf = float(box.conf[0])
                    
                    # Get class name
                    cls = int(box.cls[0])
                    class_name = model.names[cls]
                    
                    # Draw rectangle and label
                    color = colors[cls]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f'{class_name} {conf:.2f}'
                    
                    # Calculate text size for background rectangle
                    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    
                    # Draw background rectangle for text
                    cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width + 10, y1), color, -1)
                    
                    # Put text on frame
                    cv2.putText(frame, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Show the frame
            cv2.imshow("YOLOv8 Detection", frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the frame wasn't successfully read
            break

    # Release the capture object and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()