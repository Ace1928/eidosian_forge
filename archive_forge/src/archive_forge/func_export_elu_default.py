import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_elu_default() -> None:
    default_alpha = 1.0
    node = onnx.helper.make_node('Elu', inputs=['x'], outputs=['y'])
    x = np.random.randn(3, 4, 5).astype(np.float32)
    y = np.clip(x, 0, np.inf) + (np.exp(np.clip(x, -np.inf, 0)) - 1) * default_alpha
    expect(node, inputs=[x], outputs=[y], name='test_elu_default')