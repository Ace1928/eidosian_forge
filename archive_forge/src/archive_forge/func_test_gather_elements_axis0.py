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
@parameterized.expand(all_versions_for('GatherElements'))
def test_gather_elements_axis0(self, _, version) -> None:
    graph = self._make_graph([('x', TensorProto.FLOAT, (3, 3)), ('i', TensorProto.INT64, (2, 3))], [make_node('GatherElements', ['x', 'i'], ['y'], axis=0)], [])
    self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (2, 3))], opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)])