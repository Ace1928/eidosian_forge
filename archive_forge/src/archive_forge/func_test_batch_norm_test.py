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
def test_batch_norm_test(self) -> None:
    graph = self._make_graph([('x', TensorProto.FLOAT, (3, 4, 5, 6, 7)), ('scale', TensorProto.FLOAT, (4,)), ('b', TensorProto.FLOAT, (4,)), ('input_mean', TensorProto.FLOAT, (4,)), ('input_var', TensorProto.FLOAT, (4,))], [make_node('BatchNormalization', ['x', 'scale', 'b', 'input_mean', 'input_var'], ['out'], training_mode=0)], [])
    self._assert_inferred(graph, [make_tensor_value_info('out', TensorProto.FLOAT, (3, 4, 5, 6, 7))])