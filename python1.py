import numpy as np
import argparse
import time
import cv2
import os

# LABELS = open('yolo-coco/coco.names').read().split('\n')
# COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype='uint8')

# start = time.time()
# print('[INFO] Loading model')
# config_path = 'yolo-coco/yolov3.cfg'
# weights_path = 'yolo-coco/yolov3.weights'
# net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
# end = time.time()
# print("[INFO] Model loaded in {} s".format(end-start))
# ln = net.getUnconnectedOutLayersNames()
# image = cv2.imread('images/soccer.jpg')

class object_detection:

    def load_model(self):
        start = time.time()
        print('[INFO] Loading model')
        config_path = 'yolo-coco/yolov3.cfg'
        weights_path = 'yolo-coco/yolov3.weights'
        self.net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        self.LABELS = open('yolo-coco/coco.names').read().split('\n')
        self.COLORS = np.random.randint(0, 255, size=(len(self.LABELS), 3), dtype='uint8')
        end = time.time()
        print("[INFO] Model loaded in {} s".format(end-start))
        self.ln = self.net.getUnconnectedOutLayersNames()

    def detect_frame(self, image):
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)    
        start = time.time()
        layerOutputs = self.net.forward(self.ln)
        end = time.time()
        print('[INFO] YOLO took {}s to detect'.format(round(end-start, 5)))
        return layerOutputs

    def draw_box(self,layerOutputs, image):
        boxes = []
        confidences = []
        classIDs = []
        (H, W) = image.shape[:2]
        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                conf = scores[classID]
                if conf > 0.5:
                    box = detection[0:4]*np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype(int)
                    x = int(centerX - (width/2))
                    y = int(centerY - (height/2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(conf))
                    classIDs.append(classID)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = boxes[i][0], boxes[i][1]
                (w, h) = boxes[i][2], boxes[i][3]
                color = [int(c) for c in self.COLORS[classIDs[i]]]
                cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
                text = '{} : {:.4f}'.format(self.LABELS[classIDs[i]], confidences[i])
                cv2.putText(image, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return image

# od = object_detection()
# od.load_model()
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True,
# 	help="path to input image")
# args = vars(ap.parse_args())
# image = cv2.imread(args['image'])
# layerOutputs = od.detect_frame(image)
# image = od.draw_box(layerOutputs, image)
# cv2.imshow("output", image)
# cv2.imwrite("output/opt.jpg", image)