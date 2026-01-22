import itertools
import math
from typing import Sequence, Tuple, Union
import numpy as np
from onnx.reference.op_run import OpRun
def get_output_shape_explicit_padding(pads: Sequence[int], input_spatial_shape: Sequence[int], kernel_spatial_shape: Sequence[int], strides_spatial: Sequence[int], dilations: Union[Sequence[int], None]=None, ceil_mode: bool=False) -> Tuple[Sequence[int], Sequence[int]]:
    """Compute output shape according to:
    https://pytorch.org/docs/stable/generated/torch.nn.MaxPool1d.html?highlight=max+pool#torch.nn.MaxPool1d
    Pads are used to calculate output shape. Use output shape in turn to calculate the actual pads
    that are used to pad the input tensor so that computation in pool() will not cause out of bound error.
    Here is the detail. Thinking kernel as a sliding window, its size:
    sw = dilation * (kernel - 1) + 1
    l_out = (l_in + pad[0] + pad[1] - sw) / stride + 1 # (ceiled if ceil_mode is True)
    l_in_required = (l_out - 1) * stride + sw

    l_in_required is used to for computation in pool() which may be larger than padded l_in, because of ceiling.
    as an example, l_in = 3, kernel = 2, stride = 2, dilation = 1, pad = [0, 0], then
    sw = dilation * (kernel - 1) + 1 = 1 * (2 - 1) + 1 = 2
    l_out = ceil((l_in + pad[0] + pad[1] - sw) / stride + 1) = ceil((3 + 0 + 0 - 1 * (2 - 1) - 1) / 2 + 1) = 2
    l_in_required = (l_out - 1) * stride + sw = (2 - 1) * 2 + 2 = 4
    l_in_required (= 4) is not equal to l_in (= 3), so we need to pad the input tensor to l_in_required to make sure that
    the sliding window does not go out-of-bound w.r.t. input tensor. Otherwise pool() will fail.
    """
    output_spatial_shape = [0] * len(input_spatial_shape)
    pads = pads or [0] * len(input_spatial_shape) * 2
    strides_spatial = strides_spatial or [1] * len(input_spatial_shape)
    dims = len(input_spatial_shape)
    if dilations is None:
        dilations = np.ones([dims], dtype=np.int64)
    for dim in range(dims):
        dim_size = (input_spatial_shape[dim] + pads[dim] + pads[dims + dim] - dilations[dim] * (kernel_spatial_shape[dim] - 1) - 1) / strides_spatial[dim] + 1
        if ceil_mode:
            output_spatial_shape[dim] = int(np.ceil(dim_size))
        else:
            output_spatial_shape[dim] = int(np.floor(dim_size))
    pads_spatial_shape_new = pads[:]
    for dim in range(dims):
        sliding_window_size = (kernel_spatial_shape[dim] - 1) * dilations[dim] + 1
        actual_padded_input_size = (output_spatial_shape[dim] - 1) * strides_spatial[dim] + sliding_window_size
        extra_pad = actual_padded_input_size - input_spatial_shape[dim] - pads[dim] - pads[dims + dim]
        if extra_pad > 0:
            pads_spatial_shape_new[dim] += extra_pad // 2
            pads_spatial_shape_new[dims + dim] += extra_pad - extra_pad // 2
    return (output_spatial_shape, pads_spatial_shape_new)