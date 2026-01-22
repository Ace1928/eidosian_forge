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
def test_constant_value_strings(self) -> None:
    value_strings = ['o', 'n', 'n', 'x']
    graph = self._make_graph([], [make_node('Constant', [], ['y'], value_strings=value_strings)], [])
    self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.STRING, [len(value_strings)])])