import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
def pooling_output_shape_pad_lr(inputSize: int, kernelSize: int, pad_l: int, pad_r: int, stride: int, dilation: int, ceil_mode: bool):
    outputSize = div_rtn(inputSize + pad_l + pad_r - dilation * (kernelSize - 1) - 1 + (stride - 1 if ceil_mode else 0), stride) + 1
    if ceil_mode:
        if (outputSize - 1) * stride >= inputSize + pad_l:
            outputSize = outputSize - 1
    return outputSize