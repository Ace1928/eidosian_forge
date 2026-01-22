import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_Loop_2(self) -> None:
    iter_count = onnx.helper.make_tensor_value_info('iter_count', onnx.TensorProto.INT64, [])
    cond_in = onnx.helper.make_tensor_value_info('cond_in', onnx.TensorProto.BOOL, [])
    x_in = onnx.helper.make_tensor_value_info('x_in', onnx.TensorProto.FLOAT, [2, 1])
    cond_out = onnx.helper.make_tensor_value_info('cond_out', onnx.TensorProto.BOOL, [])
    x_out = onnx.helper.make_tensor_value_info('x_out', onnx.TensorProto.FLOAT, [2, 1])
    squeeze = onnx.helper.make_node('Squeeze', inputs=['x_in'], outputs=['squeeze_out'], axes=[1])
    unsqueeze = onnx.helper.make_node('Unsqueeze', inputs=['squeeze_out'], outputs=['x_out'], axes=[1])
    identity = onnx.helper.make_node('Identity', inputs=['cond_in'], outputs=['cond_out'])
    loop_body = onnx.helper.make_graph([squeeze, unsqueeze, identity], 'loop_body', [iter_count, cond_in, x_in], [cond_out, x_out])
    self._test_op_upgrade('Loop', 12, [[], '', [2, 1]], [[2, 1]], [TensorProto.INT64, TensorProto.BOOL, TensorProto.FLOAT], attrs={'body': loop_body})