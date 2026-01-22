import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_conv_with_strides() -> None:
    x = np.array([[[[0.0, 1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0, 9.0], [10.0, 11.0, 12.0, 13.0, 14.0], [15.0, 16.0, 17.0, 18.0, 19.0], [20.0, 21.0, 22.0, 23.0, 24.0], [25.0, 26.0, 27.0, 28.0, 29.0], [30.0, 31.0, 32.0, 33.0, 34.0]]]]).astype(np.float32)
    W = np.array([[[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]]).astype(np.float32)
    node_with_padding = onnx.helper.make_node('Conv', inputs=['x', 'W'], outputs=['y'], kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[2, 2])
    y_with_padding = np.array([[[[12.0, 27.0, 24.0], [63.0, 108.0, 81.0], [123.0, 198.0, 141.0], [112.0, 177.0, 124.0]]]]).astype(np.float32)
    expect(node_with_padding, inputs=[x, W], outputs=[y_with_padding], name='test_conv_with_strides_padding')
    node_without_padding = onnx.helper.make_node('Conv', inputs=['x', 'W'], outputs=['y'], kernel_shape=[3, 3], pads=[0, 0, 0, 0], strides=[2, 2])
    y_without_padding = np.array([[[[54.0, 72.0], [144.0, 162.0], [234.0, 252.0]]]]).astype(np.float32)
    expect(node_without_padding, inputs=[x, W], outputs=[y_without_padding], name='test_conv_with_strides_no_padding')
    node_with_asymmetric_padding = onnx.helper.make_node('Conv', inputs=['x', 'W'], outputs=['y'], kernel_shape=[3, 3], pads=[1, 0, 1, 0], strides=[2, 2])
    y_with_asymmetric_padding = np.array([[[[21.0, 33.0], [99.0, 117.0], [189.0, 207.0], [171.0, 183.0]]]]).astype(np.float32)
    expect(node_with_asymmetric_padding, inputs=[x, W], outputs=[y_with_asymmetric_padding], name='test_conv_with_strides_and_asymmetric_padding')