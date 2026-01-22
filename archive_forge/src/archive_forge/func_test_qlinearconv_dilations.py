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
def test_qlinearconv_dilations(self) -> None:
    graph = self._make_graph([('x', TensorProto.UINT8, (30, 4, 8, 8, 8)), ('x_scale', TensorProto.FLOAT, ()), ('x_zero_point', TensorProto.UINT8, ()), ('w', TensorProto.UINT8, (50, 4, 3, 3, 3)), ('w_scale', TensorProto.FLOAT, ()), ('w_zero_point', TensorProto.UINT8, ()), ('y_scale', TensorProto.FLOAT, ()), ('y_zero_point', TensorProto.UINT8, ())], [make_node('QLinearConv', ['x', 'x_scale', 'x_zero_point', 'w', 'w_scale', 'w_zero_point', 'y_scale', 'y_zero_point'], 'y', dilations=[1, 2, 3])], [])
    self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.UINT8, (30, 50, 6, 4, 2))])