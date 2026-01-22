import numpy as np
import onnx
from onnx import TensorProto
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
from onnx.helper import make_tensor
@staticmethod
def export_e5m2() -> None:
    node = onnx.helper.make_node('QuantizeLinear', inputs=['x', 'y_scale', 'y_zero_point'], outputs=['y'])
    x = np.array([0.0, 1.0, 2.0, 100000.0, 200.0]).astype(np.float32)
    y_scale = np.float32(2)
    y_zero_point = make_tensor('zero_point', TensorProto.FLOAT8E5M2, [1], [0.0])
    y = make_tensor('y', TensorProto.FLOAT8E5M2, [5], [0, 0.5, 1, 49152, 96])
    expect(node, inputs=[x, y_scale, y_zero_point], outputs=[y], name='test_quantizelinear_e5m2')