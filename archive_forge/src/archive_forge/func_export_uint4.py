import numpy as np
import onnx
from onnx import TensorProto
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
from onnx.helper import make_tensor
@staticmethod
def export_uint4() -> None:
    node = onnx.helper.make_node('QuantizeLinear', inputs=['x', 'y_scale', 'y_zero_point'], outputs=['y'], axis=0)
    x = np.array([[0.0, 2.5, 4.8, 8.6], [-30, -20, 6, 9], [12, 15, 16, 40]]).astype(np.float32)
    y_scale = np.asarray([2.0, 3.0, 4.0], dtype=np.float32)
    y_zero_point = make_tensor('zero_point', TensorProto.UINT4, y_scale.shape, np.ones_like(y_scale))
    y = make_tensor('y', TensorProto.UINT4, x.shape, [1, 2, 3, 5, -1, -1, 3, 4, 4, 5, 5, 11])
    expect(node, inputs=[x, y_scale, y_zero_point], outputs=[y], name='test_quantizelinear_uint4')