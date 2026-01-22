import math
import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_gelu_tanh() -> None:
    node = onnx.helper.make_node('Gelu', inputs=['x'], outputs=['y'], approximate='tanh')
    x = np.array([-1, 0, 1]).astype(np.float32)
    y = (0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))).astype(np.float32)
    expect(node, inputs=[x], outputs=[y], name='test_gelu_tanh_1')
    x = np.random.randn(3, 4, 5).astype(np.float32)
    y = (0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))).astype(np.float32)
    expect(node, inputs=[x], outputs=[y], name='test_gelu_tanh_2')