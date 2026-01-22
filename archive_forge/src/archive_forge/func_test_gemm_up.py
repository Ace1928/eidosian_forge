import contextlib
import struct
import unittest
from typing import Optional, Tuple
import numpy as np
import parameterized
import onnx.version_converter
from onnx import (
def test_gemm_up(self) -> None:
    nodes = [helper.make_node('Gemm', ['A', 'B', 'C'], ['Y'])]
    graph = helper.make_graph(nodes, 'test', [helper.make_tensor_value_info('A', TensorProto.FLOAT, (5, 5)), helper.make_tensor_value_info('B', TensorProto.FLOAT, (5, 5)), helper.make_tensor_value_info('C', TensorProto.FLOAT, (5, 5))], [helper.make_tensor_value_info('Y', TensorProto.FLOAT, (5, 5))])
    converted_model = self._converted(graph, helper.make_operatorsetid('', 1), 8)
    assert converted_model.graph.node[0].op_type == 'Gemm'
    assert converted_model.opset_import[0].version == 8