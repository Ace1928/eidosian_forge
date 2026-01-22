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
def test_adam(self) -> None:
    graph = self._make_graph([('R', TensorProto.FLOAT, ()), ('T', TensorProto.INT64, ()), ('X', TensorProto.FLOAT, (1, 2)), ('G', TensorProto.FLOAT, (1, 2)), ('V', TensorProto.FLOAT, (1, 2)), ('H', TensorProto.FLOAT, (1, 2))], [make_node('Adam', ['R', 'T', 'X', 'G', 'V', 'H'], ['X_new', 'V_new', 'H_new'], domain=AI_ONNX_PREVIEW_TRAINING_DOMAIN, alpha=0.9, beta=1.0, norm_coefficient=0.02)], [])
    infos = [make_tensor_value_info('X_new', TensorProto.FLOAT, (1, 2)), make_tensor_value_info('V_new', TensorProto.FLOAT, (1, 2)), make_tensor_value_info('H_new', TensorProto.FLOAT, (1, 2))]
    self._assert_inferred(graph, infos, opset_imports=[make_opsetid(AI_ONNX_PREVIEW_TRAINING_DOMAIN, 1), make_opsetid(ONNX_DOMAIN, 12)])