import numpy as np
import onnx
from onnx import TensorProto
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
from onnx.helper import make_tensor
@staticmethod
def export_blocked() -> None:
    node = onnx.helper.make_node('DequantizeLinear', inputs=['x', 'x_scale', 'x_zero_point'], outputs=['y'], axis=1, block_size=2)
    x = np.array([[[[3, 89], [34, 200], [74, 59]], [[5, 24], [24, 87], [32, 13]], [[5, 12], [12, 33], [65, 42]], [[245, 99], [4, 142], [121, 102]]]], dtype=np.uint8)
    x_scale = np.array([[[[3.0, 2.0], [4.0, 1.0], [2.0, 2.0]], [[5.0, 2.0], [4.0, 3.0], [5.0, 2.0]]]], dtype=np.float32)
    x_zero_point = np.array([[[[1, 0], [0, 1], [2, 20]], [[3, 2], [4, 3], [15, 2]]]], dtype=np.uint8)
    assert x_scale.shape == x_zero_point.shape
    block_axis = 1
    assert all((x.shape[i] == x_scale.shape[i] for i in range(len(x.shape)) if i != block_axis))
    assert x.shape[block_axis] % x_scale.shape[block_axis] == 0
    repeats = x.shape[block_axis] // x_scale.shape[block_axis]
    x_scale_elementwise = np.repeat(x_scale, repeats=repeats, axis=block_axis)
    x_zero_point_elementwise = np.repeat(x_zero_point, repeats=repeats, axis=block_axis)
    y = (x.astype(np.float32) - x_zero_point_elementwise.astype(np.float32)) * x_scale_elementwise
    expect(node, inputs=[x, x_scale, x_zero_point], outputs=[y], name='test_dequantizelinear_blocked')