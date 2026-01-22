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
def test_trilu_upper_zero(self) -> None:
    graph = self._make_graph([('x', TensorProto.INT64, (0, 5)), ('k', TensorProto.INT64, ())], [make_node('Trilu', ['x', 'k'], ['y'], upper=1)], [], initializer=[make_tensor('k', TensorProto.INT64, (), (5,))])
    self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.INT64, (0, 5))])