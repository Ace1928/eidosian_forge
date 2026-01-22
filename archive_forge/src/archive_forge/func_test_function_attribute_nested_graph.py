import itertools
import math
import sys
import unittest
from contextlib import redirect_stdout
from functools import wraps
from io import StringIO
from os import getenv
from textwrap import dedent
from typing import Sequence, Tuple
import numpy as np
import parameterized
import version_utils
from numpy.testing import assert_allclose
import onnx.reference.custom_element_types as custom
from onnx import (
from onnx.backend.test.case.node.roialign import get_roi_align_input_values
from onnx.checker import check_model
from onnx.defs import onnx_opset_version
from onnx.helper import (
from onnx.numpy_helper import float8e4m3_to_float32, float8e5m2_to_float32, from_array
from onnx.reference import ReferenceEvaluator
from onnx.reference.op_run import OpRun, OpRunExpand
from onnx.reference.ops import load_op
from onnx.reference.ops._op_common_indices import _get_indices, _is_out
from onnx.reference.ops._op_list import Cast_19, Celu
from onnx.reference.ops.aionnx_preview_training._op_list import Adam
from onnx.reference.ops.op_celu import _vcelu1
from onnx.reference.ops.op_col2im import (
from onnx.reference.ops.op_conv import Conv, _conv_implementation
from onnx.reference.ops_optimized import Conv as ConvOptimized
from onnx.reference.ops_optimized.op_conv_optimized import _conv_implementation_im2col
def test_function_attribute_nested_graph(self):
    opset = onnx_opset_version()
    new_domain = 'custom'
    opset_imports = [make_opsetid('', opset), make_opsetid(new_domain, 1)]
    cst1 = make_node('Constant', [], ['B1'])
    att = AttributeProto()
    att.name = 'value'
    att.ref_attr_name = 'bias1'
    att.type = AttributeProto.TENSOR
    cst1.attribute.append(att)
    cst2 = make_node('Constant', [], ['B2'])
    att = AttributeProto()
    att.name = 'value'
    att.ref_attr_name = 'bias2'
    att.type = AttributeProto.TENSOR
    cst2.attribute.append(att)
    then_out = make_tensor_value_info('B1', TensorProto.FLOAT, [None])
    else_out = make_tensor_value_info('B2', TensorProto.FLOAT, [None])
    then_body = make_graph([cst1], 'then_body', [], [then_out])
    else_body = make_graph([cst2], 'else_body', [], [else_out])
    zero = make_node('Constant', inputs=[], outputs=['zero'], value=from_array(np.array([0], dtype=np.float32)))
    mini = make_node('ReduceMin', ['X'], ['Xmin'])
    f_cond = make_node('Greater', ['Xmin', 'zero'], ['f_cond'])
    if_node = make_node('If', inputs=['f_cond'], outputs=['B'], then_branch=then_body, else_branch=else_body)
    node1 = make_node('MatMul', ['X', 'A'], ['XA'])
    node2 = make_node('Add', ['XA', 'B'], ['Y'])
    linear_regression = make_function(new_domain, 'LinearRegression', ['X', 'A'], ['Y'], [zero, mini, f_cond, if_node, node1, node2], opset_imports, ['bias1', 'bias2'])
    X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
    A = make_tensor_value_info('A', TensorProto.FLOAT, [None, None])
    Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])
    graph = make_graph([make_node('LinearRegression', ['X', 'A'], ['Y1'], domain=new_domain, bias1=make_tensor('former_B1', TensorProto.FLOAT, [1], [0.67]), bias2=make_tensor('former_B2', TensorProto.FLOAT, [1], [777])), make_node('Abs', ['Y1'], ['Y'])], 'example', [X, A], [Y])
    onnx_model = make_model(graph, opset_imports=opset_imports, functions=[linear_regression])
    check_model(onnx_model)
    sess = ReferenceEvaluator(onnx_model)
    self.assertEqual(sess.rt_nodes_[0].__class__.__name__, 'OpFunction')
    self.assertEqual(sess.rt_nodes_[0].impl_.__class__.__name__, 'ReferenceEvaluator')
    fct = sess.rt_nodes_[0].impl_
    checked = False
    for node in fct.rt_nodes_:
        if node.__class__.__name__.startswith('If'):
            if not node.has_linked_attribute:
                raise AssertionError(f'Nested node {type(node)} declares no linked attribute but a subgraph does.')
            checked = True
    if not checked:
        raise AssertionError("No node 'If' was found, has_linked_attribute could not be checked.")
    x = np.arange(6).reshape((3, 2)).astype(np.float32)
    a = np.array([1, -1], dtype=np.float32)
    result = sess.run(None, {'X': x + 1, 'A': a})[0]
    expected = np.abs(x @ a + 0.67)
    assert_allclose(expected, result)
    result = sess.run(None, {'X': x - 10, 'A': a})[0]
    expected = np.abs(x @ a + 777)
    assert_allclose(expected, result)