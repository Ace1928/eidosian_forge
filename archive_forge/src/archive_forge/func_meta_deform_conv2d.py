import functools
import torch
import torch._custom_ops
import torch.library
import torchvision.extension  # noqa: F401
@register_meta('deform_conv2d')
def meta_deform_conv2d(input, weight, offset, mask, bias, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, n_weight_grps, n_offset_grps, use_mask):
    out_height, out_width = offset.shape[-2:]
    out_channels = weight.shape[0]
    batch_size = input.shape[0]
    return input.new_empty((batch_size, out_channels, out_height, out_width))