import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_input_shape_is_NCd1d2d3_sum_weight_high_ii() -> None:
    reduction = 'sum'
    ignore_index = np.int64(10)
    node = onnx.helper.make_node('NegativeLogLikelihoodLoss', inputs=['input', 'target', 'weight'], outputs=['loss'], reduction=reduction, ignore_index=ignore_index)
    N, C = (3, 5)
    np.random.seed(0)
    input = np.random.rand(N, C).astype(np.float32)
    target = np.random.randint(0, high=C, size=N).astype(np.int64)
    target[0] = 10
    weight = np.random.rand(C).astype(np.float32)
    negative_log_likelihood_loss = compute_negative_log_likelihood_loss(input, target, weight=weight, reduction=reduction, ignore_index=ignore_index)
    expect(node, inputs=[input, target, weight], outputs=[negative_log_likelihood_loss], name='test_nllloss_NCd1d2d3_sum_weight_high_ii')