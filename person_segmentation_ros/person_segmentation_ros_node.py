#!/usr/bin/env python3

# Copyright 2024 Szymon Kwiatkowski
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy
from sensor_msgs.msg import Image
import message_filters

import cv_bridge
import numpy as np

try:
    from person_segmentation_ros.person_segmentation_ros import PersonSegmentationRos
except ImportError:
    from person_segmentation_ros import PersonSegmentationRos


class PersonSegmentationRosNode(Node):

    def __init__(self):
        super().__init__('person_segmentation_ros_node')
        distance_m = float(self.declare_parameter('distance_m', '3.0').value)
        self.input_image_height = int(self.declare_parameter('input_image_height', '320').value)
        self.input_image_width = int(self.declare_parameter('input_image_width', '320').value)
        mqtt_topic_name = str(self.declare_parameter('mqtt_topic_name', 'human-detected').value)
        model_dtype = str(self.declare_parameter('model_dtype', 'float32').value)
        onnx_model_path = str(self.declare_parameter('onnx_model_path', 'model.onnx').value)
        image_topic_name = str(self.declare_parameter('image_topic_name', '/oak/color').value)
        client_id = int(self.declare_parameter('client_id', '1').value)
        port = int(self.declare_parameter('port', '1883').value)
        broker = str(self.declare_parameter('broker', 'localhost').value)
        stereo_topic_name = str(self.declare_parameter('stereo_topic_name', '/oak/stereo').value)

        self.person_segmentation_ros = PersonSegmentationRos(
            onnx_model_path,
            mqtt_topic_name,
            model_dtype,
            distance_m,
            broker,
            port,
            client_id,
            )
        self.get_logger().info(f"onnx model path: {onnx_model_path}", once=True)
        self._bridge = cv_bridge.CvBridge()
        self._image_sub = message_filters.Subscriber(self, Image, image_topic_name)
        self._stereo_sub = message_filters.Subscriber(self, Image, stereo_topic_name)

        self._synchronizer = message_filters.ApproximateTimeSynchronizer(
            (self._image_sub, self._stereo_sub), 5, 0.01)
        self._synchronizer.registerCallback(self.onImageReceived)


    def onImageReceived(self, image_msg: Image, stereo_img: Image):
        cv_image = self._bridge.imgmsg_to_cv2(image_msg)
        # specify encoding just to make sure that everything is ok
        cv_stereo_image = self._bridge.imgmsg_to_cv2(stereo_img, '16UC1')
        if self.person_segmentation_ros.processReceivedFrames(cv_image, cv_stereo_image):
            self.get_logger().info("Message sent")


def main(args=None):
    rclpy.init(args=args)
    node = PersonSegmentationRosNode()
    try:
        rclpy.spin(node)
        node.destroy_node()
        rclpy.shutdown()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
