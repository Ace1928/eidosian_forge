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
def test_optional_construct_empty_sequence(self) -> None:
    tensor_type_proto = helper.make_tensor_type_proto(elem_type=TensorProto.INT32, shape=[1, 2, 3])
    sequence_type_proto = helper.make_sequence_type_proto(tensor_type_proto)
    optional_type_proto = helper.make_optional_type_proto(sequence_type_proto)
    optional_val_info = helper.make_value_info(name='output_sequence', type_proto=optional_type_proto)
    graph = self._make_graph([], [make_node('Optional', [], ['output_sequence'], type=sequence_type_proto)], [])
    self._assert_inferred(graph, [optional_val_info])