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
def test_conv_transpose_with_kernel_shape(self) -> None:
    graph = self._make_graph([('X', TensorProto.FLOAT, (25, 48, 16, 16)), ('W', TensorProto.FLOAT, (48, 32, None, None))], [make_node('ConvTranspose', ['X', 'W'], 'Y', kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 2, 2])], [])
    self._assert_inferred(graph, [make_tensor_value_info('Y', TensorProto.FLOAT, (25, 32, 30, 30))])