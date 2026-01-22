import os
import tempfile
import unittest
from typing import Sequence
import numpy as np
import onnx.defs
import onnx.parser
from onnx import (
def test_check_model_unsupported_output_type(self) -> None:
    N = 10
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [N])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [N])
    Z = helper.make_tensor_value_info('Z', TensorProto.BOOL, [N])
    onnx_id = helper.make_opsetid('', 6)
    node = helper.make_node('Add', ['X', 'Y'], ['Z'])
    graph = helper.make_graph([node], 'test_add_input', [X, Y], [Z])
    model = helper.make_model(graph, producer_name='test', opset_imports=[onnx_id])
    self.assertRaises(shape_inference.InferenceError, checker.check_model, model, True)