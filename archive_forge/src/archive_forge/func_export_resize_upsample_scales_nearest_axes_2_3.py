import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
from onnx.reference.ops.op_resize import _cubic_coeffs as cubic_coeffs
from onnx.reference.ops.op_resize import (
from onnx.reference.ops.op_resize import _interpolate_nd as interpolate_nd
from onnx.reference.ops.op_resize import _linear_coeffs as linear_coeffs
from onnx.reference.ops.op_resize import (
from onnx.reference.ops.op_resize import _nearest_coeffs as nearest_coeffs
@staticmethod
def export_resize_upsample_scales_nearest_axes_2_3() -> None:
    axes = [2, 3]
    node = onnx.helper.make_node('Resize', inputs=['X', '', 'scales'], outputs=['Y'], mode='nearest', axes=axes)
    data = np.array([[[[1, 2], [3, 4]]]], dtype=np.float32)
    scales = np.array([2.0, 3.0], dtype=np.float32)
    output = interpolate_nd(data, lambda x, _: nearest_coeffs(x), scale_factors=scales, axes=axes).astype(np.float32)
    expect(node, inputs=[data, scales], outputs=[output], name='test_resize_upsample_scales_nearest_axes_2_3')