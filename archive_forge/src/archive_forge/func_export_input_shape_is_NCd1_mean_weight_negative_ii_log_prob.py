import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_input_shape_is_NCd1_mean_weight_negative_ii_log_prob() -> None:
    reduction = 'mean'
    ignore_index = np.int64(-1)
    node = onnx.helper.make_node('SoftmaxCrossEntropyLoss', inputs=['x', 'y', 'w'], outputs=['z', 'log_prob'], reduction=reduction, ignore_index=ignore_index)
    N, C, dim1 = (3, 5, 6)
    np.random.seed(0)
    x = np.random.rand(N, C, dim1).astype(np.float32)
    labels = np.random.randint(0, high=C, size=(N, dim1)).astype(np.int64)
    labels[0][0] = -1
    weight = np.random.rand(C).astype(np.float32)
    loss, log_prob = softmaxcrossentropy(x, labels, weight=weight, reduction=reduction, ignore_index=ignore_index, get_log_prob=True)
    expect(node, inputs=[x, labels, weight], outputs=[loss, log_prob], name='test_sce_NCd1_mean_weight_negative_ii_log_prob')