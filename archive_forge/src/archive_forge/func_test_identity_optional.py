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
def test_identity_optional(self) -> None:
    graph = self._make_graph([('in_tensor', TensorProto.FLOAT, (2, 3, 4))], [make_node('Optional', ['in_tensor'], ['in_optional']), make_node('Identity', ['in_optional'], ['output_optional'])], [])
    tensor_type_proto = helper.make_tensor_type_proto(TensorProto.FLOAT, (2, 3, 4))
    optional_type_proto = helper.make_optional_type_proto(tensor_type_proto)
    self._assert_inferred(graph, [helper.make_value_info('in_optional', optional_type_proto), helper.make_value_info('output_optional', optional_type_proto)])