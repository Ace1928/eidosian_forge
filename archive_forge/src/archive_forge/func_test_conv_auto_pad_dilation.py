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
def test_conv_auto_pad_dilation(self) -> None:
    graph = self._make_graph([('x', TensorProto.FLOAT, (30, 4, 65, 64, 63)), ('y', TensorProto.FLOAT, (50, 4, 4, 3, 2))], [make_node('Conv', ['x', 'y'], 'z', auto_pad='SAME_UPPER', dilations=[2, 3, 4])], [])
    self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.FLOAT, (30, 50, 65, 64, 63))])