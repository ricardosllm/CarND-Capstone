from styx_msgs.msg import TrafficLight

from utils import label_map_util
from utils import visualization_utils as vis_util

import cv2
import rospy
import sys
import numpy as np
import tensorflow as tf
import os

REAL_MODEL = 'frozen_inference_graph_real.pb'
SIMULATION_SSD_MODEL = 'frozen_inference_graph_simulation_ssd.pb'
SIMULATION_RCNN_MODEL = 'frozen_inference_graph_simulation_ssd.pb'

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        curr_dir = os.path.dirname(os.path.realpath(__file__))

        path_to_ckpt = curr_dir + '/models/' + SIMULATION_SSD_MODEL

        path_to_labels = curr_dir + '/label_map.pbtxt'
        num_classes = 4

        self.detection_graph = tf.Graph()

        # Spin up frozen TensorFlow model
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            self.sess = tf.Session(graph=self.detection_graph)

        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        
        # Each box represents a part of the image where a particular object was detected.
        self.boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        
        # Score represents level of confidence for each object
        self.scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

        print("Graph loaded")

        # Load label map
        label_map = label_map_util.load_labelmap(path_to_labels)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes, use_display_name=True)

        self.category_index = label_map_util.create_category_index(categories)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        # Set unknown as default
        current_light = TrafficLight.UNKNOWN

        run = True

        if run:

            image_expanded = np.expand_dims(image, axis=0)
            
            # Detection
            with self.detection_graph.as_default():
                (boxes, scores, classes, num) = self.sess.run(
                [self.boxes, self.scores,
                self.classes, self.num_detections],
                feed_dict={self.image_tensor: image_expanded})

            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            classes = np.squeeze(classes).astype(np.int32)

            high_score = scores.argmax()

            if scores[high_score] > 0.5:
                light_color = self.category_index[classes[high_score]]['name']
                rospy.logwarn("[Classifier] {}".format(light_color))
                
                if light_color == 'Green':
                    current_light = TrafficLight.GREEN
                elif light_color == 'Red':
                    current_light = TrafficLight.RED
                elif light_color == 'Yellow':
                    current_light = TrafficLight.YELLOW
            else:
                rospy.logwarn("[Can't compute!]")


        return current_light
