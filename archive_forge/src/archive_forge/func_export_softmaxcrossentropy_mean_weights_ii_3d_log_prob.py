import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_softmaxcrossentropy_mean_weights_ii_3d_log_prob() -> None:
    reduction = 'mean'
    ignore_index = np.int64(1)
    node = onnx.helper.make_node('SoftmaxCrossEntropyLoss', inputs=['x', 'y', 'w'], outputs=['z', 'log_prob'], reduction=reduction, ignore_index=ignore_index)
    np.random.seed(0)
    x = np.random.rand(3, 5, 2).astype(np.float32)
    labels = np.random.randint(0, high=5, size=(3, 2)).astype(np.int64)
    labels[0][0] = np.int64(1)
    weights = np.array([0.2, 0.3, 0.6, 0.1, 0.5], dtype=np.float32)
    loss, log_prob = softmaxcrossentropy(x, labels, weight=weights, ignore_index=ignore_index, get_log_prob=True)
    expect(node, inputs=[x, labels, weights], outputs=[loss, log_prob], name='test_sce_mean_weight_ii_3d_log_prob')