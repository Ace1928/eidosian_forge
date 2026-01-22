import unittest
import numpy as np
from numpy.testing import assert_allclose
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.defs import onnx_opset_version
from onnx.reference import ReferenceEvaluator
from onnx.tools import update_model_dims
from onnx.tools.replace_constants import replace_initializer_by_constant_of_shape
def test_replace_initializer(self):
    dtype = np.float32
    value = np.random.randn(2, 100).astype(dtype)
    A = numpy_helper.from_array(value, name='A')
    value = np.array([1], dtype=dtype)
    C = numpy_helper.from_array(value, name='C')
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [None])
    node1 = helper.make_node('MatMul', ['X', 'A'], ['AX'])
    node2 = helper.make_node('Sub', ['AX', 'C'], ['Y'])
    graph = helper.make_graph([node1, node2], 'lr', [X], [Y], [A, C])
    model_def = helper.make_model(graph)
    x = np.array([1, 2, 4, 5, 5, 4]).astype(np.float32).reshape((3, 2))
    oinf1 = ReferenceEvaluator(model_def)
    y1 = oinf1.run(None, {'X': x})[0]
    repl = replace_initializer_by_constant_of_shape(model_def)
    node_types = {n.op_type for n in repl.graph.node}
    self.assertIn('ConstantOfShape', node_types)
    oinf2 = ReferenceEvaluator(repl)
    y1[:, :] = 3.5
    y1[0, :] = 0.5
    y2 = oinf2.run(None, {'X': x})[0]
    assert_allclose(y1, y2)