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
def test_bernoulli_with_dtype(self) -> None:
    graph = self._make_graph([('x', TensorProto.FLOAT, (2, 3, 4))], [make_node('Bernoulli', ['x'], ['out'], dtype=TensorProto.DOUBLE)], [])
    self._assert_inferred(graph, [make_tensor_value_info('out', TensorProto.DOUBLE, (2, 3, 4))])