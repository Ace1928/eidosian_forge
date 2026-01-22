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
def test_optional_tensor_has_element(self) -> None:
    tensor_type_proto = helper.make_tensor_type_proto(elem_type=TensorProto.FLOAT, shape=[2, 3, 4])
    optional_type_proto = helper.make_optional_type_proto(tensor_type_proto)
    optional_val_info = helper.make_value_info(name='sequence', type_proto=optional_type_proto)
    graph = self._make_graph([('input1', TensorProto.FLOAT, (2, 3, 4))], [make_node('Optional', ['input1'], ['sequence']), make_node('OptionalHasElement', ['sequence'], ['output'])], [])
    self._assert_inferred(graph, [optional_val_info, make_tensor_value_info('output', TensorProto.BOOL, ())])