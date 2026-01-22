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
def export_resize_upsample_sizes_nearest_not_larger() -> None:
    keep_aspect_ratio_policy = 'not_larger'
    axes = [2, 3]
    node = onnx.helper.make_node('Resize', inputs=['X', '', '', 'sizes'], outputs=['Y'], mode='nearest', axes=axes, keep_aspect_ratio_policy=keep_aspect_ratio_policy)
    data = np.array([[[[1, 2], [3, 4]]]], dtype=np.float32)
    sizes = np.array([7, 8], dtype=np.int64)
    output = interpolate_nd(data, lambda x, _: nearest_coeffs(x), output_size=sizes, axes=axes, keep_aspect_ratio_policy=keep_aspect_ratio_policy).astype(np.float32)
    expect(node, inputs=[data, sizes], outputs=[output], name='test_resize_upsample_sizes_nearest_not_larger')