import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
def conv_transpose2d_input(input: List[int], weight: List[int], bias: Optional[List[int]]=None, stride: Optional[List[int]]=None, padding: Optional[List[int]]=None, output_padding: Optional[List[int]]=None, groups: int=1, dilation: Optional[List[int]]=None) -> List[int]:
    if stride is None:
        stride = [1, 1]
    if padding is None:
        padding = [0, 0]
    if output_padding is None:
        output_padding = [0, 0]
    if dilation is None:
        dilation = [1, 1]
    has_dilation = len(dilation) > 0
    dim = len(input)
    output_size: List[int] = []
    input_batch_size_dim = 0
    weight_output_channels_dim = 1
    output_size.append(input[input_batch_size_dim])
    output_size.append(weight[weight_output_channels_dim] * groups)
    for d in range(2, dim):
        dilation_ = dilation[d - 2] if has_dilation else 1
        kernel = dilation_ * (weight[d] - 1)
        output_size.append((input[d] - 1) * stride[d - 2] - 2 * padding[d - 2] + kernel + output_padding[d - 2] + 1)
    return output_size