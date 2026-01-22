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
@parameterized.expand(all_versions_for('Transpose'))
def test_transpose_scalar(self, _, version) -> None:
    graph = self._make_graph([('X', TensorProto.FLOAT, ())], [make_node('Transpose', ['X'], ['Y'])], [])
    self._assert_inferred(graph, [make_tensor_value_info('Y', TensorProto.FLOAT, ())], opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)])