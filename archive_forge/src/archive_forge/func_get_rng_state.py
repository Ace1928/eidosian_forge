from typing import Iterable, List, Union
import torch
from .. import Tensor
from . import _lazy_call, _lazy_init, current_device, device_count
def get_rng_state(device: Union[int, str, torch.device]='cuda') -> Tensor:
    """Return the random number generator state of the specified GPU as a ByteTensor.

    Args:
        device (torch.device or int, optional): The device to return the RNG state of.
            Default: ``'cuda'`` (i.e., ``torch.device('cuda')``, the current CUDA device).

    .. warning::
        This function eagerly initializes CUDA.
    """
    _lazy_init()
    if isinstance(device, str):
        device = torch.device(device)
    elif isinstance(device, int):
        device = torch.device('cuda', device)
    idx = device.index
    if idx is None:
        idx = current_device()
    default_generator = torch.cuda.default_generators[idx]
    return default_generator.get_state()