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
def test_sequence_construct_diff_dim_size(self) -> None:
    graph = self._make_graph([('input1', TensorProto.FLOAT, (2, 3, 4)), ('input2', TensorProto.FLOAT, (2, 3, 5)), ('input3', TensorProto.FLOAT, (2, 3, 6))], [make_node('SequenceConstruct', ['input1', 'input2', 'input3'], ['output_sequence'])], [])
    self._assert_inferred(graph, [make_tensor_sequence_value_info('output_sequence', TensorProto.FLOAT, (2, 3, None))])