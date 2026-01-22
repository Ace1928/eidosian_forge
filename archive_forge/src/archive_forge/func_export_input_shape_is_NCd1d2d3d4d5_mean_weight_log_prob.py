import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_input_shape_is_NCd1d2d3d4d5_mean_weight_log_prob() -> None:
    reduction = 'mean'
    node = onnx.helper.make_node('SoftmaxCrossEntropyLoss', inputs=['x', 'y', 'w'], outputs=['z', 'log_prob'], reduction=reduction)
    N, C, dim1, dim2, dim3, dim4, dim5 = (3, 5, 6, 6, 5, 3, 4)
    np.random.seed(0)
    x = np.random.rand(N, C, dim1, dim2, dim3, dim4, dim5).astype(np.float32)
    labels = np.random.randint(0, high=C, size=(N, dim1, dim2, dim3, dim4, dim5)).astype(np.int64)
    weight = np.random.rand(C).astype(np.float32)
    loss, log_prob = softmaxcrossentropy(x, labels, weight=weight, reduction=reduction, get_log_prob=True)
    expect(node, inputs=[x, labels, weight], outputs=[loss, log_prob], name='test_sce_NCd1d2d3d4d5_mean_weight_log_prob')