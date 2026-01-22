import math
from typing import List, Optional
import torch
from torch import nn, Tensor
from .image_list import ImageList
def grid_anchors(self, grid_sizes: List[List[int]], strides: List[List[Tensor]]) -> List[Tensor]:
    anchors = []
    cell_anchors = self.cell_anchors
    torch._assert(cell_anchors is not None, 'cell_anchors should not be None')
    torch._assert(len(grid_sizes) == len(strides) == len(cell_anchors), 'Anchors should be Tuple[Tuple[int]] because each feature map could potentially have different sizes and aspect ratios. There needs to be a match between the number of feature maps passed and the number of sizes / aspect ratios specified.')
    for size, stride, base_anchors in zip(grid_sizes, strides, cell_anchors):
        grid_height, grid_width = size
        stride_height, stride_width = stride
        device = base_anchors.device
        shifts_x = torch.arange(0, grid_width, dtype=torch.int32, device=device) * stride_width
        shifts_y = torch.arange(0, grid_height, dtype=torch.int32, device=device) * stride_height
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)
        anchors.append((shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4))
    return anchors