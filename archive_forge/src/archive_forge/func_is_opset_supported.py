import os
import platform
import sys
import unittest
from typing import Any
import numpy
import version_utils
import onnx.backend.base
import onnx.backend.test
import onnx.shape_inference
import onnx.version_converter
from onnx import ModelProto
from onnx.backend.base import Device, DeviceType
from onnx.reference import ReferenceEvaluator
@classmethod
def is_opset_supported(cls, model):
    return (True, '')