import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_conv_with_autopad_same() -> None:
    x = np.array([[[[0.0, 1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0, 9.0], [10.0, 11.0, 12.0, 13.0, 14.0], [15.0, 16.0, 17.0, 18.0, 19.0], [20.0, 21.0, 22.0, 23.0, 24.0]]]]).astype(np.float32)
    W = np.array([[[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]]).astype(np.float32)
    node = onnx.helper.make_node('Conv', inputs=['x', 'W'], outputs=['y'], auto_pad='SAME_LOWER', kernel_shape=[3, 3], strides=[2, 2])
    y = np.array([[[[12.0, 27.0, 24.0], [63.0, 108.0, 81.0], [72.0, 117.0, 84.0]]]]).astype(np.float32)
    expect(node, inputs=[x, W], outputs=[y], name='test_conv_with_autopad_same')