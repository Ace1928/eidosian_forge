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
def test_slice_giant_number(self) -> None:
    graph = self._make_graph([('x', TensorProto.FLOAT, (3, 2)), ('starts', TensorProto.INT64, (2,)), ('ends', TensorProto.INT64, (2,)), ('axes', TensorProto.INT64, (2,))], [make_node('Slice', ['x', 'starts', 'ends', 'axes'], 'y')], [], initializer=[make_tensor('starts', TensorProto.INT64, (2,), (1, 0)), make_tensor('ends', TensorProto.INT64, (2,), (200, 22000)), make_tensor('axes', TensorProto.INT64, (2,), (0, 1))])
    self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (2, 2))])