import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
def pooling_output_shape(inputSize: int, kernelSize: int, pad_l: int, stride: int, dilation: int, ceil_mode: bool):
    assert stride != 0, 'stride should not be zeero'
    return pooling_output_shape_pad_lr(inputSize, kernelSize, pad_l, pad_l, stride, dilation, ceil_mode)