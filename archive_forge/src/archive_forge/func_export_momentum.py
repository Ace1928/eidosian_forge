import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
from onnx.defs import AI_ONNX_PREVIEW_TRAINING_DOMAIN
@staticmethod
def export_momentum() -> None:
    norm_coefficient = 0.001
    alpha = 0.95
    beta = 0.1
    node = onnx.helper.make_node('Momentum', inputs=['R', 'T', 'X', 'G', 'V'], outputs=['X_new', 'V_new'], norm_coefficient=norm_coefficient, alpha=alpha, beta=beta, mode='standard', domain=AI_ONNX_PREVIEW_TRAINING_DOMAIN)
    r = np.array(0.1, dtype=np.float32)
    t = np.array(0, dtype=np.int64)
    x = np.array([1.2, 2.8], dtype=np.float32)
    g = np.array([-0.94, -2.5], dtype=np.float32)
    v = np.array([1.7, 3.6], dtype=np.float32)
    x_new, v_new = apply_momentum(r, t, x, g, v, norm_coefficient, alpha, beta)
    expect(node, inputs=[r, t, x, g, v], outputs=[x_new, v_new], name='test_momentum', opset_imports=[onnx.helper.make_opsetid(AI_ONNX_PREVIEW_TRAINING_DOMAIN, 1)])