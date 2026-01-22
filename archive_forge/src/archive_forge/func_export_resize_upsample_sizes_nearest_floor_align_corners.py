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
def export_resize_upsample_sizes_nearest_floor_align_corners() -> None:
    node = onnx.helper.make_node('Resize', inputs=['X', '', '', 'sizes'], outputs=['Y'], mode='nearest', coordinate_transformation_mode='align_corners', nearest_mode='floor')
    data = np.array([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]], dtype=np.float32)
    sizes = np.array([1, 1, 8, 8], dtype=np.int64)
    output = interpolate_nd(data, lambda x, _: nearest_coeffs(x, mode='floor'), output_size=sizes, coordinate_transformation_mode='align_corners').astype(np.float32)
    expect(node, inputs=[data, sizes], outputs=[output], name='test_resize_upsample_sizes_nearest_floor_align_corners')