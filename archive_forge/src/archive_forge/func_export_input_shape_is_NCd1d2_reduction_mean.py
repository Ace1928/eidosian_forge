import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_input_shape_is_NCd1d2_reduction_mean() -> None:
    reduction = 'mean'
    node = onnx.helper.make_node('NegativeLogLikelihoodLoss', inputs=['input', 'target'], outputs=['loss'], reduction=reduction)
    N, C, dim1, dim2 = (3, 5, 6, 6)
    np.random.seed(0)
    input = np.random.rand(N, C, dim1, dim2).astype(np.float32)
    target = np.random.randint(0, high=C, size=(N, dim1, dim2)).astype(np.int64)
    negative_log_likelihood_loss = compute_negative_log_likelihood_loss(input, target, weight=None, reduction=reduction)
    expect(node, inputs=[input, target], outputs=[negative_log_likelihood_loss], name='test_nllloss_NCd1d2_reduction_mean')