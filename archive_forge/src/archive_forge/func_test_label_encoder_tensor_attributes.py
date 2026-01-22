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
@parameterized.expand(all_versions_for('LabelEncoder') if ONNX_ML else [], skip_on_empty=True)
def test_label_encoder_tensor_attributes(self, _, version) -> None:
    self.skipIf(version < 4, 'tensor attributes were introduced in ai.onnx.ml opset 4')
    key_tensor = make_tensor('keys_tensor', TensorProto.STRING, [4], ['a', 'b', 'cc', 'ddd'])
    values_tensor = make_tensor('values_tensor', TensorProto.INT64, [4], [1, 2, 3, 4])
    graph = self._make_graph([('x', TensorProto.STRING, ('M', None, 3, 12))], [make_node('LabelEncoder', ['x'], ['y'], domain=ONNX_ML_DOMAIN, keys_tensor=key_tensor, values_tensor=values_tensor, default_tensor=make_tensor('default_tensor', TensorProto.INT64, [1], [0]))], [])
    self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.INT64, ('M', None, 3, 12))], opset_imports=[make_opsetid(ONNX_ML_DOMAIN, version), make_opsetid(ONNX_DOMAIN, 11)])