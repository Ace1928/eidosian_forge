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
@parameterized.expand(all_versions_for('Reshape'))
def test_reshape_dynamic_shape_known_rank(self, _, version) -> None:
    self.skipIf(version < 14, 'Rank inference is added from Version 14')
    graph = self._make_graph([('x', TensorProto.UINT8, (2, 4, 3)), ('shape', TensorProto.INT64, (2,))], [make_node('Reshape', ['x', 'shape'], ['y'])], [])
    self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.UINT8, (None, None))], opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)])