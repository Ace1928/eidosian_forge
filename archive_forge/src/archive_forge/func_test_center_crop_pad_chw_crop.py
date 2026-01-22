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
def test_center_crop_pad_chw_crop(self):
    graph = self._make_graph([('input_data', TensorProto.FLOAT, (3, 20, 10)), ('shape', TensorProto.INT64, (2,))], [make_node('CenterCropPad', ['input_data', 'shape'], ['y'], axes=[1, 2])], [], initializer=[make_tensor('shape', TensorProto.INT64, (2,), (10, 8))])
    self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (3, 10, 8))], opset_imports=[helper.make_opsetid(ONNX_DOMAIN, 18)])