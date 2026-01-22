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
def test_mvn_function_output_shape(self) -> None:
    graph = self._make_graph([('X', TensorProto.FLOAT, (25, 48, 16, 16))], [make_node('MeanVarianceNormalization', 'X', 'Y', axes=[0, 2, 3])], [])
    self._assert_inferred(graph, [make_tensor_value_info('Y', TensorProto.FLOAT, (25, 48, 16, 16))])