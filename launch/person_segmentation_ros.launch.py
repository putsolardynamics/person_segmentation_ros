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

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import yaml


def launch_setup(context, *args, **kwargs):
    param_path = LaunchConfiguration('person_segmentation_ros_param_file').perform(context)
    if not param_path:
        param_path = PathJoinSubstitution(
            [FindPackageShare('person_segmentation_ros'), 'config', 'person_segmentation_ros.param.yaml']
        ).perform(context)

    with open(param_path, 'r', encoding="utf-8") as file:
        param_yaml = yaml.safe_load(file)

    onnx_model = LaunchConfiguration('onnx_model_path').perform(context)
    if not onnx_model:
        onnx_model = PathJoinSubstitution(
            [FindPackageShare('person_segmentation_ros'), 'resources', 'model_fp32.onnx']
        ).perform(context)

    param_yaml['onnx_model_path'] = onnx_model

    params = [{key: str(value)} for key, value in param_yaml.items()]

    person_segmentation_ros_node = Node(
        package='person_segmentation_ros',
        executable='person_segmentation_ros_node.py',
        name='person_segmentation_ros_node',
        parameters=params,
        output='screen',
        arguments=['--ros-args', '--log-level', 'info', '--enable-stdout-logs'],
        emulate_tty=True
    )

    return [
        person_segmentation_ros_node
    ]


def generate_launch_description():
    declared_arguments = []

    def add_launch_arg(name: str, default_value: str = None):
        declared_arguments.append(
            DeclareLaunchArgument(name, default_value=default_value)
        )

    add_launch_arg('person_segmentation_ros_param_file', '')
    add_launch_arg('onnx_model_path', '')

    return LaunchDescription([
        *declared_arguments,
        OpaqueFunction(function=launch_setup)
    ])
