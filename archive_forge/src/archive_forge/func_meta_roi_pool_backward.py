import functools
import torch
import torch._custom_ops
import torch.library
import torchvision.extension  # noqa: F401
@register_meta('_roi_pool_backward')
def meta_roi_pool_backward(grad, rois, argmax, spatial_scale, pooled_height, pooled_width, batch_size, channels, height, width):
    torch._check(grad.dtype == rois.dtype, lambda: f'Expected tensor for grad to have the same type as tensor for rois; but type {grad.dtype} does not equal {rois.dtype}')
    return grad.new_empty((batch_size, channels, height, width))