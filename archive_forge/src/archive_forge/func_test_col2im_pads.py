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
@parameterized.expand(all_versions_for('Col2Im'))
def test_col2im_pads(self, _, version) -> None:
    graph = self._make_graph([('input', TensorProto.FLOAT, (1, 5, 15)), ('output_shape', TensorProto.INT64, (2,)), ('kernel_shape', TensorProto.INT64, (2,))], [make_node('Col2Im', ['input', 'output_shape', 'kernel_shape'], ['output'], pads=[0, 1, 0, 1])], [], initializer=[make_tensor('output_shape', TensorProto.INT64, (2,), (5, 5)), make_tensor('kernel_shape', TensorProto.INT64, (2,), (1, 5))])
    self._assert_inferred(graph, [make_tensor_value_info('output', TensorProto.FLOAT, (1, 1, 5, 5))], opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)])