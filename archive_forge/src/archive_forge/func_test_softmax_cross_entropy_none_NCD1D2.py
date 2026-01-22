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
def test_softmax_cross_entropy_none_NCD1D2(self) -> None:
    graph = self._make_graph([('x', TensorProto.FLOAT, (2, 3, 5, 8)), ('y', TensorProto.FLOAT, (2, 5, 8))], [make_node('SoftmaxCrossEntropyLoss', ['x', 'y'], ['z'], reduction='none')], [])
    self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.FLOAT, (2, 5, 8))])