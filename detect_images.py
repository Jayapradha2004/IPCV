import cv2
import numpy as np
import os

# ===== PATHS =====
prototxt = "models/MobileNetSSD_deploy.prototxt"
model = "models/MobileNetSSD_deploy.caffemodel"

# CHANGE THIS to your dataset path
#image_folder = r"Train-20200226T103300Z-001\Train\JPEGImages"
image_folder = r"C:\Users\Jayap\image\Train-20200226T103300Z-001\Train\JPEGImages"

output_folder = "output"
os.makedirs(output_folder, exist_ok=True)

# ===== LOAD MODEL =====
net = cv2.dnn.readNetFromCaffe(prototxt, model)

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

# ===== PROCESS IMAGES =====
for img_name in os.listdir(image_folder):
    img_path = os.path.join(image_folder, img_name)

    image = cv2.imread(img_path)
    if image is None:
        continue

    (h, w) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])
            label = CLASSES[idx]

            if label == "person":
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                cv2.rectangle(image, (startX, startY), (endX, endY),
                              (0, 255, 0), 2)

                cv2.putText(image, "Person",
                            (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2)

    # Save output image
    save_path = os.path.join(output_folder, img_name)
    cv2.imwrite(save_path, image)

print("✅ Detection completed. Check output folder.")