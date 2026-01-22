from typing import Iterable, List, Union
import torch
from .. import Tensor
from . import _lazy_call, _lazy_init, current_device, device_count
def set_rng_state(new_state: Tensor, device: Union[int, str, torch.device]='cuda') -> None:
    """Set the random number generator state of the specified GPU.

    Args:
        new_state (torch.ByteTensor): The desired state
        device (torch.device or int, optional): The device to set the RNG state.
            Default: ``'cuda'`` (i.e., ``torch.device('cuda')``, the current CUDA device).
    """
    with torch._C._DisableFuncTorch():
        new_state_copy = new_state.clone(memory_format=torch.contiguous_format)
    if isinstance(device, str):
        device = torch.device(device)
    elif isinstance(device, int):
        device = torch.device('cuda', device)

    def cb():
        idx = device.index
        if idx is None:
            idx = current_device()
        default_generator = torch.cuda.default_generators[idx]
        default_generator.set_state(new_state_copy)
    _lazy_call(cb)