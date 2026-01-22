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
def test_range_rank_inference(self) -> None:
    graph = self._make_graph([('start', TensorProto.INT32, ()), ('limit', TensorProto.INT32, ()), ('delta', TensorProto.INT32, ())], [make_node('Range', ['start', 'limit', 'delta'], ['output'])], [], initializer=[make_tensor('start', TensorProto.INT32, (), (1,)), make_tensor('limit', TensorProto.INT32, (), (5,))])
    self._assert_inferred(graph, [make_tensor_value_info('output', TensorProto.INT32, (None,))])