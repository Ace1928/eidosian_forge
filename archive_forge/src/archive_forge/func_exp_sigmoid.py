import warnings
from typing import List, Optional, Union
import torch
from torchaudio.functional import fftconvolve
def exp_sigmoid(input: torch.Tensor, exponent: float=10.0, max_value: float=2.0, threshold: float=1e-07) -> torch.Tensor:
    """Exponential Sigmoid pointwise nonlinearity.
    Implements the equation:
    ``max_value`` * sigmoid(``input``) ** (log(``exponent``)) + ``threshold``

    The output has a range of [``threshold``, ``max_value``].
    ``exponent`` controls the slope of the output.

    .. devices:: CPU CUDA

    Args:
        input (Tensor): Input Tensor
        exponent (float, optional): Exponent. Controls the slope of the output
        max_value (float, optional): Maximum value of the output
        threshold (float, optional): Minimum value of the output

    Returns:
        Tensor: Exponential Sigmoid output. Shape: same as input

    """
    return max_value * torch.pow(torch.nn.functional.sigmoid(input), torch.log(torch.tensor(exponent, device=input.device, dtype=input.dtype))) + torch.tensor(threshold, device=input.device, dtype=input.dtype)