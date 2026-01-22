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
def test_constant_sparse(self) -> None:
    y_shape = [100]
    y_value = self.make_sparse(y_shape, [13, 17, 19], [3], [9, 27, 81])
    graph = self._make_graph([], [make_node('Constant', [], ['y'], sparse_value=y_value)], [])
    self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.INT64, y_shape)])