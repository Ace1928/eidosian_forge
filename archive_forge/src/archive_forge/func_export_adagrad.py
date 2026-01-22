import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
from onnx.defs import AI_ONNX_PREVIEW_TRAINING_DOMAIN
@staticmethod
def export_adagrad() -> None:
    norm_coefficient = 0.001
    epsilon = 1e-05
    decay_factor = 0.1
    node = onnx.helper.make_node('Adagrad', inputs=['R', 'T', 'X', 'G', 'H'], outputs=['X_new', 'H_new'], norm_coefficient=norm_coefficient, epsilon=epsilon, decay_factor=decay_factor, domain=AI_ONNX_PREVIEW_TRAINING_DOMAIN)
    r = np.array(0.1, dtype=np.float32)
    t = np.array(0, dtype=np.int64)
    x = np.array([1.0], dtype=np.float32)
    g = np.array([-1.0], dtype=np.float32)
    h = np.array([2.0], dtype=np.float32)
    x_new, h_new = apply_adagrad(r, t, x, g, h, norm_coefficient, epsilon, decay_factor)
    expect(node, inputs=[r, t, x, g, h], outputs=[x_new, h_new], name='test_adagrad', opset_imports=[onnx.helper.make_opsetid(AI_ONNX_PREVIEW_TRAINING_DOMAIN, 1)])