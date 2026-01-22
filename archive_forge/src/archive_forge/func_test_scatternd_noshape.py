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
@parameterized.expand(all_versions_for('ScatterND'))
def test_scatternd_noshape(self, _, version) -> None:
    graph = self._make_graph([('x', TensorProto.FLOAT, (4, 5, 6)), ('indices', TensorProto.INT64, (3, 3, 2)), ('updates', TensorProto.FLOAT, (3, 3, 6)), ('shape', TensorProto.INT64, ('M',))], [make_node('Reshape', ['x', 'shape'], ['x_reshaped']), make_node('ScatterND', ['x_reshaped', 'indices', 'updates'], ['y'])], [])
    self._assert_inferred(graph, [make_tensor_value_info('x_reshaped', TensorProto.FLOAT, None), make_tensor_value_info('y', TensorProto.FLOAT, None)], opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)])