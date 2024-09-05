import cv2

class Camera:
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)

    def capture_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            raise Exception("Failed to capture image from camera")
        return frame

    def release(self):
        self.cap.release()

def main():
    camera = Camera()
    try:
        while True:
            frame = camera.capture_frame()
            cv2.imshow("CCTV Feed", frame)

            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
