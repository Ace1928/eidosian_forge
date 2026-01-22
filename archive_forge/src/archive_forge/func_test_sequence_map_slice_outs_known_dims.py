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
def test_sequence_map_slice_outs_known_dims(self):
    body_graph = helper.make_graph(nodes=[make_node('Slice', ['x', 'starts1', 'ends1', 'axes', ''], ['y1']), make_node('Slice', ['x', 'starts2', 'ends2', 'axes', ''], ['y2'])], name='body_graph', inputs=[onnx.helper.make_tensor_value_info('x', onnx.TensorProto.FLOAT, ('H', 'W', 3))], outputs=[onnx.helper.make_tensor_value_info('y1', onnx.TensorProto.FLOAT, (10, 20, 3)), onnx.helper.make_tensor_value_info('y2', onnx.TensorProto.FLOAT, (30, 40, 3))], initializer=[make_tensor('axes', TensorProto.INT64, (2,), (0, 1)), make_tensor('starts1', TensorProto.INT64, (2,), (0, 0)), make_tensor('ends1', TensorProto.INT64, (2,), (10, 20)), make_tensor('starts2', TensorProto.INT64, (2,), (0, 0)), make_tensor('ends2', TensorProto.INT64, (2,), (30, 40))])
    graph = self._make_graph([('input1', TensorProto.FLOAT, (220, 310, 3)), ('input2', TensorProto.FLOAT, (110, 210, 3)), ('input3', TensorProto.FLOAT, (90, 110, 3))], [make_node('SequenceConstruct', ['input1', 'input2', 'input3'], ['in_sequence']), make_node('SequenceMap', ['in_sequence'], ['out_sequence1', 'out_sequence2'], body=body_graph)], [])
    self._assert_inferred(graph, [make_tensor_sequence_value_info('in_sequence', TensorProto.FLOAT, (None, None, 3)), make_tensor_sequence_value_info('out_sequence1', TensorProto.FLOAT, (10, 20, 3)), make_tensor_sequence_value_info('out_sequence2', TensorProto.FLOAT, (30, 40, 3))])