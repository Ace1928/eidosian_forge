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
def test_constant_pad_2d_opset10(self) -> None:
    graph = self._make_graph([('x', TensorProto.FLOAT, (2, 3, 4, 4))], [make_node('Pad', 'x', 'y', pads=[0, 0, 3, 1, 0, 0, 4, 2], mode='constant', value=2.0)], [])
    self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (2, 3, 11, 7))], opset_imports=[helper.make_opsetid(ONNX_DOMAIN, 10)])