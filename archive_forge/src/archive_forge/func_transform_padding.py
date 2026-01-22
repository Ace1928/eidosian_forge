import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
def transform_padding(pad_width):
    """Helper function to convert padding format for pad operator.
    """
    num_pad_values = len(pad_width)
    onnx_pad_width = [0] * num_pad_values
    start_index = 0
    end_index = int(num_pad_values / 2)
    for idx in range(0, num_pad_values):
        if idx % 2 == 0:
            onnx_pad_width[start_index] = pad_width[idx]
            start_index += 1
        else:
            onnx_pad_width[end_index] = pad_width[idx]
            end_index += 1
    return onnx_pad_width