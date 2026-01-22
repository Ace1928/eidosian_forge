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
def test_dynamicquantizelinear(self) -> None:
    graph = self._make_graph([('x', TensorProto.FLOAT, (30, 4, 5))], [make_node('DynamicQuantizeLinear', ['x'], ['y', 'y_scale', 'y_zero_point'])], [])
    self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.UINT8, (30, 4, 5)), make_tensor_value_info('y_scale', TensorProto.FLOAT, ()), make_tensor_value_info('y_zero_point', TensorProto.UINT8, ())])