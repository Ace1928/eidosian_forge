import warnings
import torch
from torch.overrides import get_default_nowrap_functions
def is_masked_tensor(a):
    """ Returns True if the input is a MaskedTensor, else False

    Args:
        a: any input

    Examples:

        >>> # xdoctest: +SKIP
        >>> from torch.masked import MaskedTensor
        >>> data = torch.arange(6).reshape(2,3)
        >>> mask = torch.tensor([[True, False, False], [True, True, False]])
        >>> mt = MaskedTensor(data, mask)
        >>> is_masked_tensor(mt)
        True
    """
    return isinstance(a, MaskedTensor)