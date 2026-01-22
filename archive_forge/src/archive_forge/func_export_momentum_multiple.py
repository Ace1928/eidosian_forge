import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
from onnx.defs import AI_ONNX_PREVIEW_TRAINING_DOMAIN
@staticmethod
def export_momentum_multiple() -> None:
    norm_coefficient = 0.001
    alpha = 0.95
    beta = 0.85
    node = onnx.helper.make_node('Momentum', inputs=['R', 'T', 'X1', 'X2', 'G1', 'G2', 'H1', 'H2'], outputs=['X1_new', 'X2_new', 'V1_new', 'V2_new'], norm_coefficient=norm_coefficient, alpha=alpha, beta=beta, mode='standard', domain=AI_ONNX_PREVIEW_TRAINING_DOMAIN)
    r = np.array(0.1, dtype=np.float32)
    t = np.array(0, dtype=np.int64)
    x1 = np.array([1.0], dtype=np.float32)
    g1 = np.array([-1.0], dtype=np.float32)
    v1 = np.array([2.0], dtype=np.float32)
    x2 = np.array([1.0, 2.0], dtype=np.float32)
    g2 = np.array([-1.0, -3.0], dtype=np.float32)
    v2 = np.array([4.0, 1.0], dtype=np.float32)
    x1_new, v1_new = apply_momentum(r, t, x1, g1, v1, norm_coefficient, alpha, beta)
    x2_new, v2_new = apply_momentum(r, t, x2, g2, v2, norm_coefficient, alpha, beta)
    expect(node, inputs=[r, t, x1, x2, g1, g2, v1, v2], outputs=[x1_new, x2_new, v1_new, v2_new], name='test_momentum_multiple', opset_imports=[onnx.helper.make_opsetid(AI_ONNX_PREVIEW_TRAINING_DOMAIN, 1)])