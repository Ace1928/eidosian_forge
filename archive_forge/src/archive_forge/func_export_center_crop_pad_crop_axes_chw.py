import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_center_crop_pad_crop_axes_chw() -> None:
    node = onnx.helper.make_node('CenterCropPad', inputs=['x', 'shape'], outputs=['y'], axes=[1, 2])
    x = np.random.randn(3, 20, 8).astype(np.float32)
    shape = np.array([10, 9], dtype=np.int64)
    y = np.zeros([3, 10, 9], dtype=np.float32)
    y[:, :, :8] = x[:, 5:15, :]
    expect(node, inputs=[x, shape], outputs=[y], name='test_center_crop_pad_crop_axes_chw')