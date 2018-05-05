#!/usr/bin/env python
import math
import rospy
import tf
import yaml

from std_msgs.msg import Int32, Header
from geometry_msgs.msg import PoseStamped, Pose, Quaternion
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier

STATE_COUNT_THRESHOLD = 1

class TLDetector(object):
    def __init__(self):
        """
        /vehicle/traffic_lights provides you with the location of the traffic
        light in 3D map space and helps you acquire an accurate ground truth
        data source for the traffic light classifier by sending the current
        color state of all traffic lights in the simulator.
        When testing on the vehicle, the color state will not be available.
        You'll need to rely on the position of the light and the camera image
        to predict it.
        """
        
        rospy.init_node('tl_detector')
        self.camera_image = None

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")

        self.config = yaml.load(config_string)
        self.light_positions = self.config['stop_line_positions']
        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """
        Callback function for camera images
        """
        
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state in [TrafficLight.RED, TrafficLight.YELLOW] else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))

        self.state_count += 1

    def create_light_site(self, x, y, z, yaw, state):
        """
        Creates a pose for the traffic light in the format required by
        get_closest_waypoint function instead of creating a new one
        """
        
        light = TrafficLight()

        light.header = Header()
        light.header.stamp = rospy.Time.now()
        light.header.frame_id = 'world'

        light.pose = PoseStamped()
        light.pose.header = Header()
        light.pose.header.stamp = rospy.Time.now()
        light.pose.header.frame_id = 'world'
        light.pose.pose.position.x = x
        light.pose.pose.position.y = y
        light.pose.pose.position.z = z

        q = tf.transformations.quaternion_from_euler(0.0, 0.0, math.pi * yaw / 180.0)
        light.pose.pose.orientation = Quaternion(*q)

        light.state = state

        return light

    def distance2d(self, x1, y1, x2, y2):
        """
        2D Euclidean distance
        """
        
        return math.sqrt((x2-x1)**2 + (y2-y1)**2)

    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints
        """

        dist = float('inf')
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        wp = 0
        for i in range(len(self.waypoints.waypoints)):
            new_dist = dl(pose.position, self.waypoints.waypoints[i].pose.pose.position)
            if new_dist < dist:
                dist = new_dist
                wp = i
        return wp

    def process_traffic_lights(self):
        """
        Finds closest visible traffic light, if one exists, and determines its
        location and color.
        """

        light = None

        if hasattr(self, 'pose') and hasattr(self, 'waypoints'):
            car_position = self.get_closest_waypoint(self.pose.pose)
            light_positions = self.light_positions

            min_distance = float('inf')
            for light_position in light_positions:
                light_candidate = self.create_light_site(light_position[0],
                                                         light_position[1],
                                                         0.0, 0.0,
                                                         TrafficLight.UNKNOWN)

                light_wp = self.get_closest_waypoint(light_candidate.pose.pose)

                # Light distance
                light_distance = self.distance2d(
                    self.waypoints.waypoints[car_position].pose.pose.position.x,
                    self.waypoints.waypoints[car_position].pose.pose.position.y,
                    self.waypoints.waypoints[light_wp].pose.pose.position.x,
                    self.waypoints.waypoints[light_wp].pose.pose.position.y
                )

                # Closest light in front of Carla
                closest_in_front_of_car = \
                  (light_wp % len(self.waypoints.waypoints)) > \
                  (car_position % len(self.waypoints.waypoints))

                within_range = (light_distance < 100) and \
                               (light_distance < min_distance)

                if closest_in_front_of_car and within_range:
                    light = light_candidate
                    closest_light_wp = light_wp
                    min_distance = light_distance

            if light:
                if self.camera_image is None:
                    rospy.logwarn('[Detector] No image')
                    return False
                else:
                    self.camera_image.encoding = "rgb8"
                    cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "rgb8")
                    state = self.light_classifier.get_classification(cv_image)

                    if state == TrafficLight.UNKNOWN and self.last_state:
                        state = self.last_state

                rospy.logwarn(
                    '[TD] Traffic light id {} in sight, color state: {}'
                    .format(closest_light_wp, state)
                )
                light_wp = closest_light_wp
            else:
                light_wp = -1
                state = TrafficLight.UNKNOWN

        else:
            light_wp = -1
            state = TrafficLight.RED

        return light_wp, state


if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
