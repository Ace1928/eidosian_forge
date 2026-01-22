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
def test_unsqueeze_scalar(self) -> None:
    graph = self._make_graph([('x', TensorProto.FLOAT, ()), ('axes', TensorProto.INT64, ())], [make_node('Unsqueeze', ['x', 'axes'], 'y')], [], initializer=[make_tensor('axes', TensorProto.INT64, (), (-1,))])
    self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (1,))])