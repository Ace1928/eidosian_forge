import itertools
import random
import struct
import unittest
from typing import Any, List, Tuple
import numpy as np
import parameterized
import pytest
import version_utils
from onnx import (
from onnx.reference.op_run import to_array_extended
def test_unknown_dimensions(self) -> None:
    graph = helper.make_graph([helper.make_node('Add', ['X', 'Y_Initializer'], ['Z'])], 'test', [helper.make_tensor_value_info('X', TensorProto.FLOAT, [None])], [helper.make_tensor_value_info('Z', TensorProto.FLOAT, [None])], [helper.make_tensor('Y_Initializer', TensorProto.FLOAT, [1], [1])], doc_string=None)
    model = helper.make_model(graph)
    checker.check_model(model)
    graph_str = helper.printable_graph(graph)
    self.assertIn('X[FLOAT, ?]', graph_str)