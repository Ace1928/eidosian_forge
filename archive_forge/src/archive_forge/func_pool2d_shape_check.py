import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
def pool2d_shape_check(input: List[int], kH: int, kW: int, dH: int, dW: int, padH: int, padW: int, dilationH: int, dilationW: int, nInputPlane: int, inputHeight: int, inputWidth: int, outputHeight: int, outputWidth: int):
    ndim = len(input)
    nOutputPlane = nInputPlane
    assert kW > 0 and kH > 0
    assert dW > 0 and dH > 0
    assert dilationH > 0 and dilationW > 0
    valid_dims = input[1] != 0 and input[2] != 0
    assert ndim == 3 and input[0] != 0 and valid_dims or (ndim == 4 and valid_dims and (input[3] != 0))
    assert kW // 2 >= padW and kH // 2 >= padH
    assert outputWidth >= 1 and outputHeight >= 1