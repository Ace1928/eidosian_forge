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
def test_depth_to_space(self) -> None:
    b = 10
    graph = self._make_graph([('x', TensorProto.FLOAT, (2, 300, 10, 10))], [make_node('DepthToSpace', ['x'], ['z'], blocksize=b, mode='DCR')], [])
    self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.FLOAT, (2, 3, 100, 100))])