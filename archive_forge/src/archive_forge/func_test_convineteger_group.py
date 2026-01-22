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
def test_convineteger_group(self) -> None:
    graph = self._make_graph([('x', TensorProto.INT8, (30, 4, 8, 8, 8)), ('y', TensorProto.INT8, (4, 1, 8, 8, 8))], [make_node('ConvInteger', ['x', 'y'], 'z', group=4)], [])
    self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.INT32, (30, 4, 1, 1, 1))])