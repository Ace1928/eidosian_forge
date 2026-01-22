from __future__ import annotations
import itertools
import unittest
from typing import Any, Sequence
import numpy as np
import pytest
from parameterized import parameterized
import onnx.shape_inference
from onnx import (
from onnx.defs import (
from onnx.helper import (
from onnx.parser import parse_graph
def test_parse_data_with_undefined_tensor_type(self) -> None:
    model = helper.make_model(graph=helper.make_graph(name='graph_with_undefined_type', inputs=[], outputs=[helper.make_tensor_value_info('y', TensorProto.FLOAT, shape=None)], nodes=[make_node('ConstantOfShape', ['x'], ['y'])], initializer=[numpy_helper.from_array(np.array([4, 3], dtype=np.int64), name='x')]))
    model.graph.initializer[0].data_type = TensorProto.UNDEFINED
    self.assertRaises(onnx.shape_inference.InferenceError, onnx.shape_inference.infer_shapes, model, strict_mode=True)
    inferred_model = onnx.shape_inference.infer_shapes(model)
    self.assertFalse(inferred_model.graph.output[0].type.tensor_type.HasField('shape'))
    graph = self._make_graph([('x', TensorProto.UINT8, (1, 0, 0)), ('shape', TensorProto.INT64, (3,))], [make_node('Reshape', ['x', 'shape'], ['y'], allowzero=1)], [], initializer=[make_tensor('shape', TensorProto.INT64, (3,), (0, 1, 1))])
    self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.UINT8, (0, 1, 1))])