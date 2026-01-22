import numpy as np
import onnx
from onnx import TensorProto
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
from onnx.helper import make_tensor
@staticmethod
def export_blocked_asymmetric() -> None:
    node = onnx.helper.make_node('QuantizeLinear', inputs=['x', 'y_scale', 'y_zero_point'], outputs=['y'], axis=1, block_size=2)
    x = np.array([[6.0, 12.0, 50.0, 5.0], [1.0, 8.0, 4.0, 5.0], [0.0, 20.0, 10.0, 4.0]], dtype=np.float32)
    y_scale = np.array([[1.5, 2.5], [3.0, 4.9], [5.1, 6.9]], dtype=np.float32)
    y_zero_point = np.array([[0, 1], [1, 0], [2, 3]], dtype=np.uint8)
    assert y_scale.shape == y_zero_point.shape
    block_axis = 1
    assert all((x.shape[i] == y_scale.shape[i] for i in range(len(x.shape)) if i != block_axis))
    assert x.shape[block_axis] % y_scale.shape[block_axis] == 0
    repeats = x.shape[block_axis] // y_scale.shape[block_axis]
    y_scale_elementwise = np.repeat(y_scale, repeats=repeats, axis=block_axis)
    y_zero_point_elementwise = np.repeat(y_zero_point, repeats=repeats, axis=block_axis)
    y = np.rint(x / y_scale_elementwise + y_zero_point_elementwise).astype(np.uint8)
    expect(node, inputs=[x, y_scale, y_zero_point], outputs=[y], name='test_quantizelinear_blocked_asymmetric')