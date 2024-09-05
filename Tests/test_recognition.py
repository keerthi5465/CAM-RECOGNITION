import unittest
import cv2
from cctv_object_recognition.recognition import ObjectRecognizer

class TestObjectRecognizer(unittest.TestCase):
    def test_recognition(self):
        recognizer = ObjectRecognizer("yolov3.weights", "yolov3.cfg", "coco.names")
        frame = cv2.imread("test_image.jpg")
        recognized_objects = recognizer.recognize(frame)
        self.assertIsInstance(recognized_objects, list)

if __name__ == '__main__':
    unittest.main()

