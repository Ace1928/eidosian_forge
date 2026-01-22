import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
def pad_impl(data, raw_pads, mode, constant_values=0.0, axes=None):
    input_rank = data.ndim
    if axes is None:
        axes = list(range(input_rank))
    else:
        axes = [axis if axis >= 0 else axis + input_rank for axis in axes]
    num_axes = len(axes)
    if num_axes * 2 != raw_pads.size:
        raise Exception('The number of elements in raw_pads should be 2 * num_axes')
    pad_width = []
    for _ in range(input_rank):
        pad_width += [[0, 0]]
    for i in range(num_axes):
        axis = axes[i]
        if axis < 0:
            axis = input_rank + axis
        pad_width[axis] = [raw_pads[i], raw_pads[i + num_axes]]
    if mode == 'constant':
        y = np.pad(data, pad_width=pad_width, mode=mode, constant_values=constant_values)
        return y
    y = np.pad(data, pad_width=pad_width, mode=mode)
    return y