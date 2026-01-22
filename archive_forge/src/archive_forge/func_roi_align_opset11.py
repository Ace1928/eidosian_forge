import sys
import warnings
import torch
from torch.onnx import symbolic_opset11 as opset11
from torch.onnx.symbolic_helper import parse_args
@parse_args('v', 'v', 'f', 'i', 'i', 'i', 'i')
def roi_align_opset11(g, input, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio, aligned):
    batch_indices = _process_batch_indices_for_roi_align(g, rois)
    rois = _process_rois_for_roi_align(g, rois)
    if aligned:
        warnings.warn('ROIAlign with aligned=True is only supported in opset >= 16. Please export with opset 16 or higher, or use aligned=False.')
    sampling_ratio = _process_sampling_ratio_for_roi_align(g, sampling_ratio)
    return g.op('RoiAlign', input, rois, batch_indices, spatial_scale_f=spatial_scale, output_height_i=pooled_height, output_width_i=pooled_width, sampling_ratio_i=sampling_ratio)