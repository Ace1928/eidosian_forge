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
def test_maxunpool_shape_without_output_shape(self) -> None:
    graph = self._make_graph([('xT', TensorProto.FLOAT, (1, 1, 2, 2)), ('xI', TensorProto.FLOAT, (1, 1, 2, 2))], [make_node('MaxUnpool', ['xT', 'xI'], 'Y', kernel_shape=[2, 2], strides=[2, 2])], [])
    self._assert_inferred(graph, [make_tensor_value_info('Y', TensorProto.FLOAT, (1, 1, 4, 4))])