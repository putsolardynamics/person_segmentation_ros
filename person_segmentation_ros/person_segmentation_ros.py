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

from pathlib import Path
import numpy as np
import onnxruntime as ort
import paho.mqtt.client as mqtt
import cv2

class PersonSegmentationRos:
    def __init__(self,
                 onnx_model_path: str,
                 topic_name: str,
                 dtype, distance: float,
                 broker: str,
                 port: int,
                 client_id: int) -> None:
        self.session = ort.InferenceSession(onnx_model_path)
        self.output_name = self.session.get_outputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        self.input_name = self.session.get_inputs()[0].name
        self.input_height, self.input_width = input_shape[2:]
        self.dtype = dtype
        self._uint16_max = 2**16 - 1
        self._detection_state = False
        self._threshold = .65
        self._min_camera_range = .7 # min range which is 0.0
        self._max_camera_range = 12 # max range which would be equal to uint16 max

        self.detection_distance = np.clip(
           ((distance - self._min_camera_range) / self._max_camera_range) * self._uint16_max, 
           10,
           self._uint16_max)
        
        print(self.detection_distance, distance)

        self._client = mqtt.Client(self.get_subscriber_id(client_id, topic_name))
        self._client.connect(broker, port, 60)
        self.topic_name = topic_name
    
    def infer(self, frame: np.ndarray):
        frame = np.asarray(cv2.resize(frame, (self.input_width, self.input_height)))
        input_data = self.preprocess(frame, dtype=self.dtype)
        outputs = self.session.run([self.output_name], {self.input_name: input_data})[0]
        predicted_mask = np.where(outputs < self._threshold , 0, 255).astype(np.uint8)
        return np.squeeze(predicted_mask)
    
    @staticmethod
    def get_subscriber_id(client_id: int, topic: str) -> str:
        """Get specifically formatted client id"""
        return f"subscriber-{topic}-{client_id}"
    
    def sendMqttMessage(self, message: bool):
        self._client.publish(self.topic_name, payload=message, qos=0, retain=False)
    
    def processReceivedFrames(self, image: np.ndarray, stereo_image: np.ndarray) -> bool:
       segmentation_mask = self.infer(image)
      #  segmentation_mask = cv2.erode(segmentation_mask, np.ones((3, 3)), iterations=4)
      #  segmentation_mask = cv2.dilate(segmentation_mask, np.ones((5, 5)), iterations=5)
       image = cv2.resize(image, (self.input_width, self.input_height))
       stereo_image = np.array(cv2.resize(stereo_image, (self.input_width, self.input_height)), dtype=np.uint16)
       processed_img = self.stereoSegmentedFusion(mask=segmentation_mask, stereo=stereo_image)
       # Save results in local directory
      #  cv2.imwrite('mask_applied.jpg', np.array(processed_img / 256, dtype=np.uint8))
      #  cv2.imwrite('mask.jpg', segmentation_mask)
      #  cv2.imwrite('image.jpg', image)
      #  cv2.imwrite('stereo.jpg', np.array(stereo_image / 256, dtype=np.uint8))
       return self.detectionAction(processed_img)

    def stereoSegmentedFusion(self, mask: np.ndarray, stereo: np.ndarray) -> np.ndarray:
       stereo_cp = stereo.copy()
       stereo[mask<=0]=self._uint16_max
       stereo = cv2.add(stereo_cp, stereo)
       return np.array(stereo, dtype=np.uint16)
    
    def detectionAction(self, detection_img: np.ndarray) -> bool:
       print(detection_img.min())
       current_state = detection_img.min() < self.detection_distance
       if (current_state == self._detection_state):
          return False
       
       self.sendMqttMessage(str(current_state))
       self._detection_state = current_state
       return True

    @staticmethod
    def preprocess(image: np.ndarray, dtype=np.float32):
        if dtype==np.uint8:
          image = image.astype(dtype)
        else:
          image = (image / 255).astype(dtype)
        image = np.transpose(image, (2, 0, 1))

        return image[np.newaxis, ...]
