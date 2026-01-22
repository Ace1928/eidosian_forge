from typing import Tuple
import numpy as np
from onnx.reference.op_run import OpRun
@staticmethod
def roi_align_forward(output_shape: Tuple[int, int, int, int], bottom_data, spatial_scale, height: int, width: int, sampling_ratio, bottom_rois, num_roi_cols: int, top_data, mode, half_pixel: bool, batch_indices_ptr):
    n_rois = output_shape[0]
    channels = output_shape[1]
    pooled_height = output_shape[2]
    pooled_width = output_shape[3]
    for n in range(n_rois):
        index_n = n * channels * pooled_width * pooled_height
        offset_bottom_rois = n * num_roi_cols
        roi_batch_ind = batch_indices_ptr[n]
        offset = 0.5 if half_pixel else 0.0
        roi_start_w = bottom_rois[offset_bottom_rois + 0] * spatial_scale - offset
        roi_start_h = bottom_rois[offset_bottom_rois + 1] * spatial_scale - offset
        roi_end_w = bottom_rois[offset_bottom_rois + 2] * spatial_scale - offset
        roi_end_h = bottom_rois[offset_bottom_rois + 3] * spatial_scale - offset
        roi_width = roi_end_w - roi_start_w
        roi_height = roi_end_h - roi_start_h
        if not half_pixel:
            roi_width = max(roi_width, 1.0)
            roi_height = max(roi_height, 1.0)
        bin_size_h = roi_height / pooled_height
        bin_size_w = roi_width / pooled_width
        roi_bin_grid_h = int(sampling_ratio) if sampling_ratio > 0 else int(np.ceil(roi_height / pooled_height))
        roi_bin_grid_w = int(sampling_ratio) if sampling_ratio > 0 else int(np.ceil(roi_width / pooled_width))
        count = int(max(roi_bin_grid_h * roi_bin_grid_w, 1))
        pre_calc = [PreCalc() for i in range(roi_bin_grid_h * roi_bin_grid_w * pooled_width * pooled_height)]
        RoiAlign.pre_calc_for_bilinear_interpolate(height, width, pooled_height, pooled_width, roi_bin_grid_h, roi_bin_grid_w, roi_start_h, roi_start_w, bin_size_h, bin_size_w, roi_bin_grid_h, roi_bin_grid_w, pre_calc)
        for c in range(channels):
            index_n_c = index_n + c * pooled_width * pooled_height
            offset_bottom_data = int((roi_batch_ind * channels + c) * height * width)
            pre_calc_index = 0
            for ph in range(pooled_height):
                for pw in range(pooled_width):
                    index = index_n_c + ph * pooled_width + pw
                    output_val = 0.0
                    if mode == 'avg':
                        for _iy in range(roi_bin_grid_h):
                            for _ix in range(roi_bin_grid_w):
                                pc = pre_calc[pre_calc_index]
                                output_val += pc.w1 * bottom_data[offset_bottom_data + pc.pos1] + pc.w2 * bottom_data[offset_bottom_data + pc.pos2] + pc.w3 * bottom_data[offset_bottom_data + pc.pos3] + pc.w4 * bottom_data[offset_bottom_data + pc.pos4]
                                pre_calc_index += 1
                        output_val /= count
                    else:
                        max_flag = False
                        for _iy in range(roi_bin_grid_h):
                            for _ix in range(roi_bin_grid_w):
                                pc = pre_calc[pre_calc_index]
                                val = max(pc.w1 * bottom_data[offset_bottom_data + pc.pos1], pc.w2 * bottom_data[offset_bottom_data + pc.pos2], pc.w3 * bottom_data[offset_bottom_data + pc.pos3], pc.w4 * bottom_data[offset_bottom_data + pc.pos4])
                                if not max_flag:
                                    output_val = val
                                    max_flag = True
                                else:
                                    output_val = max(output_val, val)
                                pre_calc_index += 1
                    top_data[index] = output_val