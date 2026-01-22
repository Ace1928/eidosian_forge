import contextlib
import struct
import unittest
from typing import Optional, Tuple
import numpy as np
import parameterized
import onnx.version_converter
from onnx import (
def test_dropout_down(self) -> None:
    nodes = [helper.make_node('Dropout', ['data'], ['output'])]
    graph = helper.make_graph(nodes, 'test', [helper.make_tensor_value_info('data', TensorProto.FLOAT, (5, 5))], [helper.make_tensor_value_info('output', TensorProto.FLOAT, (5, 5))])
    converted_model = self._converted(graph, helper.make_operatorsetid('', 8), 1)
    assert converted_model.graph.node[0].op_type == 'Dropout'
    assert converted_model.opset_import[0].version == 1