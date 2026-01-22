import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_softmaxcrossentropy_mean_no_weights_ii_4d() -> None:
    reduction = 'mean'
    ignore_index = np.int64(2)
    node = onnx.helper.make_node('SoftmaxCrossEntropyLoss', inputs=['x', 'y'], outputs=['z'], reduction=reduction, ignore_index=ignore_index)
    np.random.seed(0)
    x = np.random.rand(3, 5, 2, 7).astype(np.float32)
    labels = np.random.randint(0, high=5, size=(3, 2, 7)).astype(np.int64)
    labels[0][0][0] = np.int64(2)
    sce = softmaxcrossentropy(x, labels, reduction=reduction, ignore_index=ignore_index)
    expect(node, inputs=[x, labels], outputs=[sce], name='test_sce_mean_no_weight_ii_4d')