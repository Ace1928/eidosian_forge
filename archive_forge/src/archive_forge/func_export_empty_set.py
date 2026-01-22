import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_empty_set() -> None:
    shape = [2, 0, 4]
    keepdims = 1
    reduced_shape = [2, 1, 4]
    node = onnx.helper.make_node('ReduceL2', inputs=['data', 'axes'], outputs=['reduced'], keepdims=keepdims)
    data = np.array([], dtype=np.float32).reshape(shape)
    axes = np.array([1], dtype=np.int64)
    reduced = np.array(np.zeros(reduced_shape, dtype=np.float32))
    expect(node, inputs=[data, axes], outputs=[reduced], name='test_reduce_l2_empty_set')