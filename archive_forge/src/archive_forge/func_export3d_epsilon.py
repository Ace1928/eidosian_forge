import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export3d_epsilon() -> None:
    epsilon = 0.1
    X = np.random.randn(2, 3, 5).astype(np.float32)

    def case(axis: int) -> None:
        normalized_shape = calculate_normalized_shape(X.shape, axis)
        W = np.random.randn(*normalized_shape).astype(np.float32)
        B = np.random.randn(*normalized_shape).astype(np.float32)
        Y, mean, inv_std_dev = _layer_normalization(X, W, B, axis, epsilon)
        node = onnx.helper.make_node('LayerNormalization', inputs=['X', 'W', 'B'], outputs=['Y', 'Mean', 'InvStdDev'], axis=axis, epsilon=epsilon)
        if axis < 0:
            name = f'test_layer_normalization_3d_axis_negative_{-axis}_epsilon'
        else:
            name = f'test_layer_normalization_3d_axis{axis}_epsilon'
        expect(node, inputs=[X, W, B], outputs=[Y, mean, inv_std_dev], name=name)
    for i in range(len(X.shape)):
        case(i)
        case(i - len(X.shape))