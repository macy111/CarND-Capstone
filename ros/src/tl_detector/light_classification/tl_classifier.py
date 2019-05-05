from styx_msgs.msg import TrafficLight
from yolo import YOLO
import PIL.Image as Image
import numpy as np
from PIL import ImageFile
from matplotlib import pyplot as plt
import cv2
import os
import tensorflow as  tf
ImageFile.LOAD_TRUNCATED_IMAGES = True

class TLClassifier(object):
    def __init__(self):
        abs_path = os.path.dirname(os.path.abspath(__file__))
        self.model = YOLO(model_path=os.path.join(abs_path,'trained_weights.h5'))
        self.graph = tf.get_default_graph()
        self.count = 0
        

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        image = Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))  
        with self.graph.as_default():
            img , _, pred_classes = self.model.detect_image(image)
            #path = "/home/student/test/"+str(self.count)+".jpg"
            #self.count += 1
            #img.save(path)
        pred_classes = list(set(pred_classes))
        print(pred_classes)
        if len(pred_classes) == 1:
            if pred_classes[0] == "red":
                return TrafficLight.RED
            elif pred_classes[0] == "yellow":
                return TrafficLight.YELLOW
            elif pred_classes[0] == "green":
                return TrafficLight.GREEN
        return TrafficLight.UNKNOWN
