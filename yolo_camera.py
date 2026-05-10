from ultralytics import YOLO
import cv2
import pyttsx3

# Load model
model = YOLO("yolov8n.pt")

# Voice engine
engine = pyttsx3.init()

# Start camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame)

    # 🔊 VOICE ALERT (INSERTED HERE)
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            if label == "person":
                engine.say("Person detected ahead")
                engine.runAndWait()

    # Draw boxes
    annotated_frame = results[0].plot()

    # Show output
    cv2.imshow("YOLO Camera Detection", annotated_frame)

    # Exit on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()