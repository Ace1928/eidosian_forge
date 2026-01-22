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
def test_GLU(self) -> None:
    graph = self._make_graph([('x', TensorProto.FLOAT, (5, 6, 7))], [make_node('Split', ['x'], ['y', 'z'], axis=1, num_outputs=2), make_node('Sigmoid', ['z'], ['a']), make_node('Mul', ['y', 'a'], ['b'])], [])
    self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (5, 3, 7)), make_tensor_value_info('z', TensorProto.FLOAT, (5, 3, 7)), make_tensor_value_info('a', TensorProto.FLOAT, (5, 3, 7)), make_tensor_value_info('b', TensorProto.FLOAT, (5, 3, 7))])