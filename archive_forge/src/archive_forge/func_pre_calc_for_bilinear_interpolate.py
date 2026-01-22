from typing import Tuple
import numpy as np
from onnx.reference.op_run import OpRun
@staticmethod
def pre_calc_for_bilinear_interpolate(height: int, width: int, pooled_height: int, pooled_width: int, iy_upper: int, ix_upper: int, roi_start_h, roi_start_w, bin_size_h, bin_size_w, roi_bin_grid_h: int, roi_bin_grid_w: int, pre_calc):
    pre_calc_index = 0
    for ph in range(pooled_height):
        for pw in range(pooled_width):
            for iy in range(iy_upper):
                yy = roi_start_h + ph * bin_size_h + (iy + 0.5) * bin_size_h / roi_bin_grid_h
                for ix in range(ix_upper):
                    xx = roi_start_w + pw * bin_size_w + (ix + 0.5) * bin_size_w / roi_bin_grid_w
                    x = xx
                    y = yy
                    if y < -1.0 or y > height or x < -1.0 or (x > width):
                        pc = pre_calc[pre_calc_index]
                        pc.pos1 = 0
                        pc.pos2 = 0
                        pc.pos3 = 0
                        pc.pos4 = 0
                        pc.w1 = 0
                        pc.w2 = 0
                        pc.w3 = 0
                        pc.w4 = 0
                        pre_calc_index += 1
                        continue
                    y = max(y, 0)
                    x = max(x, 0)
                    y_low = int(y)
                    x_low = int(x)
                    if y_low >= height - 1:
                        y_high = y_low = height - 1
                        y = y_low
                    else:
                        y_high = y_low + 1
                    if x_low >= width - 1:
                        x_high = x_low = width - 1
                        x = x_low
                    else:
                        x_high = x_low + 1
                    ly = y - y_low
                    lx = x - x_low
                    hy = 1.0 - ly
                    hx = 1.0 - lx
                    w1 = hy * hx
                    w2 = hy * lx
                    w3 = ly * hx
                    w4 = ly * lx
                    pc = PreCalc()
                    pc.pos1 = y_low * width + x_low
                    pc.pos2 = y_low * width + x_high
                    pc.pos3 = y_high * width + x_low
                    pc.pos4 = y_high * width + x_high
                    pc.w1 = w1
                    pc.w2 = w2
                    pc.w3 = w3
                    pc.w4 = w4
                    pre_calc[pre_calc_index] = pc
                    pre_calc_index += 1