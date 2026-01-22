import decimal
import numpy as np
import torch
from torch import nn
from torch.autograd import Function
from ...utils import logging
def batch_frexp(inputs, max_bit=31):
    """
    Decompose the scaling factor into mantissa and twos exponent.

    Args:
        scaling_factor (`torch.Tensor`):
            Target scaling factor to decompose.

    Returns:
        ``Tuple(torch.Tensor, torch.Tensor)`: mantisa and exponent
    """
    shape_of_input = inputs.size()
    inputs = inputs.view(-1)
    output_m, output_e = np.frexp(inputs.cpu().numpy())
    tmp_m = []
    for m in output_m:
        int_m_shifted = int(decimal.Decimal(m * 2 ** max_bit).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP))
        tmp_m.append(int_m_shifted)
    output_m = np.array(tmp_m)
    output_e = float(max_bit) - output_e
    return (torch.from_numpy(output_m).to(inputs.device).view(shape_of_input), torch.from_numpy(output_e).to(inputs.device).view(shape_of_input))