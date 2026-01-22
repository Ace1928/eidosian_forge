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
def test_reshape_static_shape_allowzero(self, _, version) -> None:
    self.skipIf(version < 14, 'allowzero is added from Version 14')
    graph = self._make_graph([('x', TensorProto.UINT8, (1, 0, 0)), ('shape', TensorProto.INT64, (3,))], [make_node('Reshape', ['x', 'shape'], ['y'], allowzero=1)], [], initializer=[make_tensor('shape', TensorProto.INT64, (3,), (0, 1, 1))])
    self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.UINT8, (0, 1, 1))], opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)])