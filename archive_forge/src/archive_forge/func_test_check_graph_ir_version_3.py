import os
import tempfile
import unittest
from typing import Sequence
import numpy as np
import onnx.defs
import onnx.parser
from onnx import (
def test_check_graph_ir_version_3(self) -> None:
    ctx = checker.C.CheckerContext()
    ctx.ir_version = 3
    ctx.opset_imports = {'': onnx.defs.onnx_opset_version()}
    lex_ctx = checker.C.LexicalScopeContext()

    def check_ir_version_3(g: GraphProto) -> None:
        checker.check_graph(g, ctx, lex_ctx)
    node = helper.make_node('Relu', ['X'], ['Y'], name='test')
    graph = helper.make_graph([node], 'test', [helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 2])], [helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 2])])
    check_ir_version_3(graph)
    graph.initializer.extend([self._sample_float_tensor])
    graph.initializer[0].name = 'no-exist'
    self.assertRaises(checker.ValidationError, check_ir_version_3, graph)
    graph.initializer[0].name = 'X'
    check_ir_version_3(graph)