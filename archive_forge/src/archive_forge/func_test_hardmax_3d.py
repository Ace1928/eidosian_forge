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
def test_hardmax_3d(self) -> None:
    graph = self._make_graph([('x', TensorProto.FLOAT, (4, 5, 6))], [make_node('Hardmax', ['x'], 'z')], [])
    self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.FLOAT, (4, 5, 6))])