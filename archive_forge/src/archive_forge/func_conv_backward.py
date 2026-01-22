import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional
from .expanded_weights_utils import \
def conv_backward(func, ctx, grad_output):

    def weight_grad_sample(weight):
        if batch_size < THRESHOLD and groups == 1:
            return conv_group_weight_grad_sample(ctx.input, grad_output, weight_shape, stride, padding, dilation, batch_size, func)
        else:
            return conv_unfold_weight_grad_sample(ctx.input, grad_output, weight_shape, kernel_size, stride, padding, dilation, groups, func)

    def expand(param):
        if isinstance(param, int):
            return conv_picker(func, (param,), (param, param), (param, param, param))
        else:
            return param

    def calc_total_padding(func, was_same, padding, dilation, kernel_size):
        if was_same:
            all_padding = int_padding_for_string_padding(func, 'same', dilation, kernel_size)
            total_padding = tuple((all_padding[i] + all_padding[i - 1] for i in range(len(all_padding) - 1, -1, -2)))
            return total_padding
        else:
            return tuple((2 * pad for pad in padding))
    weight_shape = ctx.weight.shape
    stride, padding, dilation, groups = (expand(ctx.stride), expand(ctx.padding), expand(ctx.dilation), ctx.groups)
    kernel_size = []
    for i in range(2, conv_picker(func, 3, 4, 5)):
        kernel_size.append(weight_shape[i])
    batch_size = ctx.batch_size
    results: List[Optional[torch.Tensor]] = []
    results.append(None)
    results.append(None)
    total_padding = calc_total_padding(func, ctx.was_same_padding, padding, dilation, kernel_size)
    if ctx.input_required_grad:
        output_padding = []
        input_dims = conv_picker(func, 1, 2, 3)
        for i in range(input_dims):
            input_dim = ctx.orig_input_shape[2 + i]
            output_padding.append((total_padding[i] + input_dim - (kernel_size[i] * dilation[i] - dilation[i] + 1)) % stride[i])
        weight_ = unpack_expanded_weight_or_tensor(ctx.weight)
        transpose_func = conv_picker(func, F.conv_transpose1d, F.conv_transpose2d, F.conv_transpose3d)
        out = transpose_func(grad_output, weight_, None, stride, padding, tuple(output_padding), groups, dilation)
        if ctx.was_same_padding:
            for i in range(len(total_padding)):
                out = torch.narrow(out, 2 + i, total_padding[i] // 2, ctx.orig_input_shape[2 + i])
        results.append(out)
    else:
        results.append(None)
    results = results + [None] * 6
    set_grad_sample_if_exists(ctx.weight, weight_grad_sample)
    set_grad_sample_if_exists(ctx.bias, lambda _: grad_output.reshape(*grad_output.shape[:2], -1).sum(dim=2))
    return tuple(results)