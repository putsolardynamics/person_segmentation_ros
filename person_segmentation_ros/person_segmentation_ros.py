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
    def __init__(self, onnx_model_path: str, topic_name: str, dtype, distance: float) -> None:
        self.session = ort.InferenceSession(onnx_model_path)
        self.output_name = self.session.get_outputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        self.input_name = self.session.get_inputs()[0].name
        self.input_height, self.input_width = input_shape[2:]
        self.dtype = dtype
        self.detection_distance = distance

        self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
        self._client.connect('localhost', 1883, 60)
        self.topic_name = topic_name
    
    def infer(self, frame: np.ndarray):
        frame = np.asarray(cv2.resize(frame, (self.input_width, self.input_height)))
        input_data = self.preprocess(frame, dtype=self.dtype)
        outputs = self.session.run([self.output_name], {self.input_name: input_data})[0]
        predicted_mask = np.where(outputs < 0.5, 0, 255).astype(np.uint8)
        return predicted_mask
    
    def sendMqttMessage(self, message: bool):
        self._client.publish(self.topic_name, payload=message, qos=0, retain=False)
    
    def processReceivedFrames(self, image: np.ndarray, stereo_image: np.ndarray):
       segmentation_mask = self.infer(image)
       cv2.imshow('mask', segmentation_mask)
       # TODO: implement logic
       return

    @staticmethod
    def preprocess(image: np.ndarray, dtype=np.float32):
        if dtype==np.uint8:
          image = image.astype(dtype)
        else:
          image = (image / 255).astype(dtype)
        image = np.transpose(image, (2, 0, 1))

        return image[np.newaxis, ...]
