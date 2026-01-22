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
def test_convinteger(self) -> None:
    graph = self._make_graph([('x', TensorProto.UINT8, (3, 4, 5, 6, 7)), ('y', TensorProto.UINT8, (5, 4, 2, 4, 3))], [make_node('ConvInteger', ['x', 'y'], 'z', pads=[0, 1, 1, 0, 0, 1], dilations=[1, 2, 2], strides=[1, 1, 2])], [])
    self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.INT32, (3, 5, 4, 1, 3))])