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
@unittest.skip('Issue #5960')
def test_quantizelinear_invalid_output_dtype(self) -> None:
    graph = self._make_graph([('x', TensorProto.FLOAT, (3, 4, 5)), ('y_scale', TensorProto.FLOAT, ())], [make_node('QuantizeLinear', ['x', 'y_scale'], ['y'], output_dtype=TensorProto.FLOAT16)], [])
    self.assertRaises(onnx.shape_inference.InferenceError, self._inferred, graph)