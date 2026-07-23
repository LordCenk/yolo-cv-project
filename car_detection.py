import cv2
import os
import yaml
import numpy as np
from yaml.loader import SafeLoader

class YOLO_Pred():
    def __init__(self,onnx_model,data_yaml):
        # Load YAML
        with open('data.yaml',mode='r') as f:
            data_yaml = yaml.load(f,Loader=SafeLoader)
            
        self.labels = data_yaml['names']
        
        # Load YOLO model
        self.yolo = cv2.dnn.readNetFromONNX('./Model/weights/best.onnx')
        self.yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def predictions(self,image):
 
        row, col, d = image.shape
        # Convert image into square matrix array
        max_rc = max(row,col)
        input_image = np.zeros((max_rc,max_rc,3),dtype=np.uint8)
        input_image[0:row,0:col] = image
        INPUT_WH_YOLO = 640
        blob = cv2.dnn.blobFromImage(input_image,1/255,(INPUT_WH_YOLO,INPUT_WH_YOLO),swapRB=True,crop=False)
        self.yolo.setInput(blob)
        preds = self.yolo.forward()

        # Non maximum supression (Extract detection based on the confidence and probablity score)
        detections = preds[0]
        boxes = []
        confidences = []
        classes = []

        # Get the height and width of the input image
        image_w , image_h = input_image.shape[:2]
        x_factor = image_w/INPUT_WH_YOLO
        y_factor = image_h/INPUT_WH_YOLO

        for i in range(len(detections)):
            row = detections[i]
            confidence = row[4]
            if confidence > 0.4:
                class_score = row[5:].max()
                class_id = row[5:].argmax()
                if class_score > 0.25:
                    cx, cy, w, h = row[0:4]
                    # Constructing the bounding box
                    left = int((cx - 0.5*w)*x_factor)
                    top = int((cy - 0.5*h)*y_factor)
                    width = int(w*x_factor)
                    height = int(h*y_factor)

                    box = np.array([left,top,width,height])

                    # append all the values into list

                    confidences.append(confidence)
                    boxes.append(box)
                    classes.append(class_id)

        # Cleaning the data and converting into required format
        boxes_np = np.array(boxes).tolist()
        confidences_np = np.array(confidences).tolist()

        # Non maximum supression for boxes_np and confidences_np
        index = cv2.dnn.NMSBoxes(boxes_np,confidences_np,0.25,0.45).flatten()

        # Draw the bounding box
        for i in index:
            x,y,w,h = boxes_np[i]
            bb_conf = int(confidences_np[i]*100)
            classes_id = classes[i]
            class_name = self.labels[classes_id]
            text = f'{class_name}: {bb_conf}%'
            cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.rectangle(image,(x,y-30),(x+w,y),(255,255,255),-1)
            cv2.putText(image,text,(x,y-10),cv2.FONT_HERSHEY_PLAIN,0.7,(0,0,0),1)

        return index
