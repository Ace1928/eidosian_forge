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
@parameterized.expand(all_versions_for('StringSplit'))
def test_string_split_nested(self, _, version) -> None:
    substrings = make_tensor_value_info('substrings', TensorProto.STRING, (2, 4, 3, None))
    length = make_tensor_value_info('length', TensorProto.INT64, (2, 4, 3))
    graph = self._make_graph([('x', TensorProto.STRING, (2, 4, 3))], [make_node('StringSplit', ['x'], ['substrings', 'length'], maxsplit=2)], [substrings, length])
    self._assert_inferred(graph, [substrings, length], opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)])