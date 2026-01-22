import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_hardsigmoid_default() -> None:
    default_alpha = 0.2
    default_beta = 0.5
    node = onnx.helper.make_node('HardSigmoid', inputs=['x'], outputs=['y'])
    x = np.random.randn(3, 4, 5).astype(np.float32)
    y = np.clip(x * default_alpha + default_beta, 0, 1)
    expect(node, inputs=[x], outputs=[y], name='test_hardsigmoid_default')