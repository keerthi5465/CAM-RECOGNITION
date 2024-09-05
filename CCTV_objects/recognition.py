import cv2
import numpy as np

class ObjectRecognizer:
    def __init__(self, model_path, config_path, labels_path):
        self.net = cv2.dnn.readNet(model_path, config_path)
        with open(labels_path, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]

    def recognize(self, frame):
        blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        layer_names = self.net.getUnconnectedOutLayersNames()
        detections = self.net.forward(layer_names)
        return self._process_detections(detections, frame)

    def _process_detections(self, detections, frame):
        recognized_objects = []
        height, width = frame.shape[:2]

        for detection in detections:
            for obj in detection:
                scores = obj[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(obj[0] * width)
                    center_y = int(obj[1] * height)
                    w = int(obj[2] * width)
                    h = int(obj[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    recognized_objects.append({
                        "label": self.labels[class_id],
                        "confidence": float(confidence),
                        "bbox": (x, y, w, h)
                    })

                    # Drawing the bounding box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, f"{self.labels[class_id]}: {confidence:.2f}", 
                                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return recognized_objects
