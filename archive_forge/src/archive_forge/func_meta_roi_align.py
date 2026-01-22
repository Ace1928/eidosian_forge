import functools
import torch
import torch.library
import torchvision.extension  # noqa: F401
@register_meta('roi_align')
def meta_roi_align(input, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio, aligned):
    torch._check(rois.size(1) == 5, lambda: 'rois must have shape as Tensor[K, 5]')
    torch._check(input.dtype == rois.dtype, lambda: f'Expected tensor for input to have the same type as tensor for rois; but type {input.dtype} does not equal {rois.dtype}')
    num_rois = rois.size(0)
    _, channels, height, width = input.size()
    return input.new_empty((num_rois, channels, pooled_height, pooled_width))