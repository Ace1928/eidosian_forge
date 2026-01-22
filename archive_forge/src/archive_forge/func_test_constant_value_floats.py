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
def test_constant_value_floats(self) -> None:
    value_floats = [1.0, 1.1, 1.2]
    graph = self._make_graph([], [make_node('Constant', [], ['y'], value_floats=value_floats)], [])
    self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, [len(value_floats)])])