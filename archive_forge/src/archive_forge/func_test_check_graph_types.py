import os
import tempfile
import unittest
from typing import Sequence
import numpy as np
import onnx.defs
import onnx.parser
from onnx import (
def test_check_graph_types(self) -> None:
    node_div = helper.make_node('Div', ['X', 'Y'], ['Z'], name='test_div')
    node_identity = helper.make_node('Identity', ['Z'], ['W'], name='test_identity')
    graph = helper.make_graph([node_div, node_identity], 'test', [helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 2]), helper.make_tensor_value_info('Y', TensorProto.BOOL, [1, 2])], [helper.make_tensor_value_info('W', TensorProto.FLOAT, [1, 2])])
    model = helper.make_model(graph, producer_name='test')
    self.assertRaises(shape_inference.InferenceError, checker.check_model, model, True)
    checker.check_graph(graph)
    graph = helper.make_graph([node_div, node_identity], 'test', [helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 2]), helper.make_tensor_value_info('Y', TensorProto.INT32, [1, 2])], [helper.make_tensor_value_info('W', TensorProto.FLOAT, [1, 2])])
    model = helper.make_model(graph, producer_name='test')
    self.assertRaises(shape_inference.InferenceError, checker.check_model, model, True)
    checker.check_graph(graph)