import rospy
from pid import PID
from yaw_controller import YawController
from math import tanh

GAS_DENSITY = 2.858
ONE_MPH = 0.44704

class Controller(object):
	def __init__(self, wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle):
		# TODO: Implement
		self.pid_controller = PID(3.0, 0.5, 5.0)
		self.yaw_controller = YawController(
			wheel_base = wheel_base,
			steer_ratio = steer_ratio,
			min_speed = min_speed,
			max_lat_accel = max_lat_accel,
			max_steer_angle = max_steer_angle
		)
		self.prev_throttle = 0
		self.prev_time = None

	def control(self, twist_cmd, current_velocity):
		# TODO: Change the arg, kwarg list to suit your needs
		# Return throttle, brake, steer
		
		# corner case
		if not self.prev_time:
			self.prev_time = rospy.get_time()
			return 0, 0, 0

		# get current params
		## from /twist_cmd
		target_linear_velocity = twist_cmd.twist.linear.x
		target_angular_velocity = twist_cmd.twist.angular.z
		## from /current_velocity
		current_linear_velocity = current_velocity.twist.linear.x
		current_angular_velocity = current_velocity.twist.angular.z
		## from /dbw_enabled

		# pid controller
		delta_v = target_linear_velocity - current_linear_velocity
		delta_t = float(rospy.get_time() - self.prev_time)
		control = self.pid_controller.step(
			error = delta_v,
			sample_time = delta_t
		)

		# get throttle according to pid controller
		throttle = tanh(control)
		if throttle - self.prev_throttle > 0.1:
			throttle = self.prev_throttle + 0.1
		elif throttle - self.prev_throttle < -0.1:
			throttle = self.prev_throttle - 0.1

		# get steering through yaw_controller
		steering = self.yaw_controller.get_steering(
			linear_velocity = target_linear_velocity,
			angular_velocity = target_angular_velocity,
			current_velocity = current_linear_velocity)
		
		# update params
		self.prev_time = rospy.get_time()
		self.prev_throttle = throttle

		return throttle, 0., steering