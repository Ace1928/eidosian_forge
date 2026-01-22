import unittest
import numpy as np
from numpy.testing import assert_allclose
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.defs import onnx_opset_version
from onnx.reference import ReferenceEvaluator
from onnx.tools import update_model_dims
from onnx.tools.replace_constants import replace_initializer_by_constant_of_shape
def test_replace_constant_graph(self):
    value = np.array([0], dtype=np.float32)
    zero = numpy_helper.from_array(value, name='zero')
    X = helper.make_tensor_value_info('X', onnx.TensorProto.FLOAT, [None, None])
    Y = helper.make_tensor_value_info('Y', onnx.TensorProto.FLOAT, [None])
    rsum = helper.make_node('ReduceSum', ['X'], ['rsum'])
    cond = helper.make_node('Greater', ['rsum', 'zero'], ['cond'])
    then_out = helper.make_tensor_value_info('then_out', onnx.TensorProto.FLOAT, None)
    then_cst = numpy_helper.from_array(np.array([1] * 129).astype(np.float32))
    then_const_node = helper.make_node('Constant', inputs=[], outputs=['then_out'], value=then_cst, name='cst1')
    then_body = helper.make_graph([then_const_node], 'then_body', [], [then_out])
    else_out = helper.make_tensor_value_info('else_out', onnx.TensorProto.FLOAT, None)
    else_cst = numpy_helper.from_array(np.array([-1] * 129).astype(np.float32))
    else_const_node = helper.make_node('Constant', inputs=[], outputs=['else_out'], value=else_cst, name='cst2')
    else_body = helper.make_graph([else_const_node], 'else_body', [], [else_out])
    if_node = onnx.helper.make_node('If', ['cond'], ['Y'], then_branch=then_body, else_branch=else_body)
    graph = helper.make_graph([rsum, cond, if_node], 'if', [X], [Y], [zero])
    onnx_model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', onnx_opset_version())])
    self.assertNotIn('ConstantOfShape', str(onnx_model))
    x = np.ones((3, 2), dtype=np.float32)
    oinf1 = ReferenceEvaluator(onnx_model)
    y1 = oinf1.run(None, {'X': x})[0]
    repl = replace_initializer_by_constant_of_shape(onnx_model)
    self.assertIn('ConstantOfShape', str(repl))
    oinf2 = ReferenceEvaluator(repl)
    y2 = oinf2.run(None, {'X': x})[0]
    y1 = y1.copy()
    y1[:] = 0.5
    assert_allclose(y1, y2)