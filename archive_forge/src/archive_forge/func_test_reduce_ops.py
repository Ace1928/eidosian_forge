import unittest
import automatic_conversion_test_base
import numpy as np
import parameterized
import onnx
from onnx import helper
@parameterized.parameterized.expand(['ReduceL1', 'ReduceL2', 'ReduceLogSum', 'ReduceLogSumExp', 'ReduceMean', 'ReduceMax', 'ReduceMin', 'ReduceProd', 'ReduceSum', 'ReduceSumSquare'])
def test_reduce_ops(self, op) -> None:
    axes = helper.make_tensor('b', onnx.TensorProto.INT64, dims=[3], vals=np.array([0, 1, 2]))
    self._test_op_downgrade(op, from_opset=13, input_shapes=[[3, 4, 5], [3]], output_shapes=[[1, 1, 1]], input_types=[onnx.TensorProto.FLOAT, onnx.TensorProto.INT64], initializer=[axes])