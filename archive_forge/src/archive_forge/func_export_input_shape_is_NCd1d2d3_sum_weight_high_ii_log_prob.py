import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_input_shape_is_NCd1d2d3_sum_weight_high_ii_log_prob() -> None:
    reduction = 'sum'
    ignore_index = np.int64(10)
    node = onnx.helper.make_node('SoftmaxCrossEntropyLoss', inputs=['x', 'y', 'w'], outputs=['z', 'log_prob'], reduction=reduction, ignore_index=ignore_index)
    N, C = (3, 5)
    np.random.seed(0)
    x = np.random.rand(N, C).astype(np.float32)
    labels = np.random.randint(0, high=C, size=N).astype(np.int64)
    labels[0] = 10
    weight = np.random.rand(C).astype(np.float32)
    loss, log_prob = softmaxcrossentropy(x, labels, weight=weight, reduction=reduction, ignore_index=ignore_index, get_log_prob=True)
    expect(node, inputs=[x, labels, weight], outputs=[loss, log_prob], name='test_sce_NCd1d2d3_sum_weight_high_ii_log_prob')