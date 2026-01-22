import functools
import torch
import torch._custom_ops
import torch.library
import torchvision.extension  # noqa: F401
@register_meta('roi_pool')
def meta_roi_pool(input, rois, spatial_scale, pooled_height, pooled_width):
    torch._check(rois.size(1) == 5, lambda: 'rois must have shape as Tensor[K, 5]')
    torch._check(input.dtype == rois.dtype, lambda: f'Expected tensor for input to have the same type as tensor for rois; but type {input.dtype} does not equal {rois.dtype}')
    num_rois = rois.size(0)
    channels = input.size(1)
    out_size = (num_rois, channels, pooled_height, pooled_width)
    return (input.new_empty(out_size), torch.empty(out_size, device='meta', dtype=torch.int32))