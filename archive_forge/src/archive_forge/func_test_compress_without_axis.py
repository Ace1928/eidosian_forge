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
def test_compress_without_axis(self) -> None:
    graph = self._make_graph([('input', TensorProto.INT64, (2, 'N', 3, 4)), ('condition', TensorProto.BOOL, (None,))], [make_node('Compress', ['input', 'condition'], ['output'])], [])
    self._assert_inferred(graph, [make_tensor_value_info('output', TensorProto.INT64, (None,))])