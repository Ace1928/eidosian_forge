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
def test_affinegrid_2d(self) -> None:
    N, C, H, W = (2, 3, 4, 5)
    graph = self._make_graph([('theta', TensorProto.FLOAT, (N, 2, 3)), ('size', TensorProto.INT64, (4,))], [make_node('AffineGrid', ['theta', 'size'], ['grid'], align_corners=1)], [], initializer=[make_tensor('size', TensorProto.INT64, (4,), (N, C, H, W))])
    self._assert_inferred(graph, [make_tensor_value_info('grid', TensorProto.FLOAT, (N, H, W, 2))])