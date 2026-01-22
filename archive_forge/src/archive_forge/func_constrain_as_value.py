from typing import Optional
import torch
def constrain_as_value(symbol, min: Optional[int]=None, max: Optional[int]=None):
    """
    Add min/max constraint on the intermediate symbol at tracing time. If called in eager mode,
    it will still check if the input value is within the specified range.
    """
    torch.sym_constrain_range(symbol, min=min, max=max)