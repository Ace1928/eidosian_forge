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
def test_lp_norm(self) -> None:
    graph = self._make_graph([('x', TensorProto.FLOAT, (3, 4, 5, 6, 7))], [make_node('LpNormalization', ['x'], ['out'])], [])
    self._assert_inferred(graph, [make_tensor_value_info('out', TensorProto.FLOAT, (3, 4, 5, 6, 7))])