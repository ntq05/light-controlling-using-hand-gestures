from utils.NeuralNetwork import NeuralNetwork
from utils.data_loader import ProcessData
from utils.detector import HandLandMarksDetector
from utils.extract_class import label_dict_from_config_file

import torch

import numpy as np

import cv2

class LightGesture:
    def __init__(self, model_path):
        self.height = 720
        self.width = 1280

        self.detector = HandLandMarksDetector()
        self.status_text = None
        self.signs = label_dict_from_config_file("../Dataset/hand_gesture.yaml")
        self.classifier = NeuralNetwork()
        self.classifier.load_state_dict(torch.load(model_path))

        self.classifier.eval()

        self.light1 = False
        self.light2 = False
        self.light3 = False

        self.light_simulation = self.light_device

    def light_device(self, img, lights):
        height, width, _ = img.shape
        rect_height = int(0.15 * height)
        rect_width = width

        white_rect = np.ones((rect_height, rect_width, 3), dtype=np.uint8) * 255

        cv2.rectangle(white_rect, (0, 0), (rect_width, rect_height), (0, 0, 255), 2)

        circle_radius = int(0.45 * rect_height)
        circle1_center = (int(rect_width * 0.25), int(rect_height / 2))
        circle2_center = (int(rect_width * 0.5), int(rect_height / 2))
        circle3_center = (int(rect_width * 0.75), int(rect_height / 2))

        on_color = (0, 255, 255)
        off_color = (0, 0, 0)
        colors = [off_color, on_color]

        circle_centers = [circle1_center, circle2_center, circle3_center]
        for cc, light in zip(circle_centers, lights):
            color = colors[int(light)]
            cv2.circle(white_rect, cc, circle_radius, color, -1)
            img = np.vstack((img, white_rect))

        return img
    
    def run(self):
        cam = cv2.VideoCapture(0)
        cam.set(3,1280)
        cam.set(4,720)
        while cam.isOpened():
            _, frame = cam.read()

            hand, img = self.detector.detecthand(frame)
            if len(hand) != 0:
                with torch.no_grad():
                    hand_landmark = torch.from_numpy(np.array(hand[0], dtype=np.float32).flatten()).unsqueeze(0)
                    hand_landmark = ProcessData(hand_landmark).process()
                    hand_landmark = torch.FloatTensor(hand_landmark).unsqueeze(0)
                    class_number = self.classifier.predict(hand_landmark).item()

                    if class_number != -1:
                        self.status_text = self.signs[class_number]
                        if self.status_text == "light1":
                            self.light1 = True
                            self.light2 = False
                            self.light3 = False
                        elif self.status_text == "light2":
                            self.light2 = True
                            self.light1 = False
                            self.light3 = False
                        elif self.status_text == "light3":
                            self.light3 = True
                            self.light1 = False
                            self.light2 = False
                        elif self.status_text == "turn_off":
                            self.light1 = False
                            self.light2 = False
                            self.light3 = False
                        elif self.status_text == "turn_on":
                            self.light1 = True
                            self.light2 = True
                            self.light3 = True

                    else:
                        self.status_text = "Unknown gesture"

            else:
                self.status_text = "No hand detected"

            img = self.light_simulation(img, [self.light1, self.light2, self.light3])
            cv2.putText(img, self.status_text, (5,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.namedWindow('window', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('window', 1920, 1080)
            cv2.imshow("window",img)
            key = cv2.waitKey(1)
            if key == ord("q"):
                break
        cv2.destroyAllWindows()

if __name__ == "__main__":
    model_path = "../Model/best_model.pth"
    light_gesture = LightGesture(model_path)
    light_gesture.run()