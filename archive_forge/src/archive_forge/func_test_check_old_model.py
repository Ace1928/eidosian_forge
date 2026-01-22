import os
import tempfile
import unittest
from typing import Sequence
import numpy as np
import onnx.defs
import onnx.parser
from onnx import (
def test_check_old_model(self) -> None:
    node = helper.make_node('Pad', ['X'], ['Y'], paddings=(0, 0, 0, 0))
    graph = helper.make_graph([node], 'test', [helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 2])], [helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 2])])
    onnx_id = helper.make_opsetid('', 1)
    model = helper.make_model(graph, producer_name='test', opset_imports=[onnx_id])
    checker.check_model(model)