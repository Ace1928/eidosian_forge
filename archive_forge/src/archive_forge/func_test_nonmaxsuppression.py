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
def test_nonmaxsuppression(self) -> None:
    graph = self._make_graph([('boxes', TensorProto.FLOAT, (1, 3, 4)), ('scores', TensorProto.FLOAT, (1, 5, 3))], [make_node('NonMaxSuppression', ['boxes', 'scores'], ['y'])], [])
    self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.INT64, (None, 3))])