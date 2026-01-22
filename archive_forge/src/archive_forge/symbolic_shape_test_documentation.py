import unittest
from typing import List, Optional
import onnx.shape_inference
from onnx import ModelProto, TensorProto, TensorShapeProto, ValueInfoProto, helper
from onnx.helper import make_model, make_tensor_value_info
Get shape from tensor_type or sparse_tensor_type according to given name