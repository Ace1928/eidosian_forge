import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_input_shape_is_NCd1_mean_weight_negative_ii() -> None:
    reduction = 'mean'
    ignore_index = np.int64(-1)
    node = onnx.helper.make_node('NegativeLogLikelihoodLoss', inputs=['input', 'target', 'weight'], outputs=['loss'], reduction=reduction, ignore_index=ignore_index)
    N, C, dim1 = (3, 5, 6)
    np.random.seed(0)
    input = np.random.rand(N, C, dim1).astype(np.float32)
    target = np.random.randint(0, high=C, size=(N, dim1)).astype(np.int64)
    target[0][0] = -1
    weight = np.random.rand(C).astype(np.float32)
    negative_log_likelihood_loss = compute_negative_log_likelihood_loss(input, target, weight=weight, reduction=reduction, ignore_index=ignore_index)
    expect(node, inputs=[input, target, weight], outputs=[negative_log_likelihood_loss], name='test_nllloss_NCd1_mean_weight_negative_ii')