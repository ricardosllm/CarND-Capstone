from styx_msgs.msg import TrafficLight

import numpy as np
import os
import sys
import tensorflow as tf
from collections import defaultdict
from io import StringIO
from utilities import label_map_util
from utilities import visualization_utils as vis_util
import time

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier

        self.current_light = TrafficLight.UNKNOWN

        path = os.path.dirname(os.path.realpath(__file__))
     
        self.simulation = True  # Toggle for real (false)

        # Path to frozen graphs
        if self.simulation is True:
            CKPT = path+'/graphs/Faster-RCNN-Sim/frozen_inference_graph.pb'
        else:
            CKPT = path+'/graphs/Faster-RCNN-Real/frozen_inference_graph.pb'

        PATH_TO_LABELS = path+'/graphs/label_map.pbtxt'
        NUM_CLASSES = 4

        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)

        # Need to review this below
        self.image_np_deep = None
        self.detection_graph = tf.Graph()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with self.detection_graph.as_default():
            graph_def = tf.GraphDef()

            with tf.gfile.GFile(CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(graph_def, name='')

            self.sess = tf.Session(graph=self.detection_graph, config=config)

        # Input and output
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

        # Bounding boxes
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

        print("Graph is ready")

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction

        run = True # Toggle to disable

        if run is True:
            image_expanded = np.expand_dims(image, axis=0)

            # Detection
            with self.detection_graph.as_default():
                (boxes, scores, classes, num) = self.sess.run(
                    [self.detection_boxes, self.detection_scores, 
                    self.detection_classes, self.num_detections],
                    feed_dict={self.image_tensor: image_np_expanded})

        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)

        # Traffic light distance

        min_score_thresh = .50
        for i in range(boxes.shape[0]):
            if scores is None or scores[i] > min_score_thresh:
                
                class_name = self.category_index[classes[i]]['name']
                # class_id = self.category_index[classes[i]]['id']  # if needed

                print('{}'.format(class_name))

                # Traffic lights
                self.current_light = TrafficLight.UNKNOWN

                if class_name == 'Red':
                    self.current_light = TrafficLight.RED
                elif class_name == 'Green':
                    self.current_light = TrafficLight.GREEN
                elif class_name == 'Yellow':
                    self.current_light = TrafficLight.YELLOW

                fx =  1345.200806
                fy =  1353.838257
                perceived_width_x = (boxes[i][3] - boxes[i][1]) * 800
                perceived_width_y = (boxes[i][2] - boxes[i][0]) * 600

                # ymin, xmin, ymax, xmax = box
                # depth_prime = (width_real * focal) / perceived_width
                # traffic light is 4 feet long and 1 foot wide?
                perceived_depth_x = ((1 * fx) / perceived_width_x)
                perceived_depth_y = ((3 * fy) / perceived_width_y )

                estimated_distance = round((perceived_depth_x + perceived_depth_y) / 2)

        # Visualization of detection results
        vis_util.visualize_boxes_and_labels_on_image_array(
            image, boxes, classes, scores,
            self.category_index,
            use_normalized_coordinates=True,
            line_thickness=8)
        
    # Visualization of topic output
    self.image_np_deep = image

    return self.current_light