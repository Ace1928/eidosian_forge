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
@parameterized.expand(all_versions_for('StringConcat'))
def test_stringconcat(self, _, version) -> None:
    graph = self._make_graph([('x', TensorProto.STRING, (2, 3, 4)), ('y', TensorProto.STRING, (2, 3, 4))], [make_node('StringConcat', ['x', 'y'], 'z')], [])
    self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.STRING, (2, 3, 4))], opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)])