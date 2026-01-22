from __future__ import annotations
import unittest
from shape_inference_test import TestShapeInferenceHelper
import onnx.parser
from onnx import TensorProto
from onnx.helper import make_node, make_tensor, make_tensor_value_info
def test_expand_symbolic_input(self) -> None:
    graph = self._make_graph([('x', TensorProto.INT32, (3, 1, 2)), ('y', TensorProto.INT32, (1, 4, 2))], [make_node('Shape', ['y'], ['shape']), make_node('Expand', ['x', 'shape'], ['z'])], [])
    self._assert_inferred(graph, [make_tensor_value_info('shape', TensorProto.INT64, (3,)), make_tensor_value_info('z', TensorProto.INT32, (3, 4, 2))], data_prop=True)