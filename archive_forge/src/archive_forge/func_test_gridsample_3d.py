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
def test_gridsample_3d(self) -> None:
    graph = self._make_graph([('x', TensorProto.FLOAT, (1, 1, 3, 3, 3)), ('grid', TensorProto.INT64, (1, 3, 2, 3, 3))], [make_node('GridSample', ['x', 'grid'], ['y'], mode='nearest', padding_mode='border', align_corners=1)], [])
    self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (1, 1, 3, 2, 3))])