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
def test_split_negative_axis(self) -> None:
    graph = self._make_graph([('x', TensorProto.FLOAT, (2, 4))], [make_node('Split', ['x'], ['y', 'z'], axis=-1, num_outputs=2)], [])
    self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (2, 2)), make_tensor_value_info('z', TensorProto.FLOAT, (2, 2))])