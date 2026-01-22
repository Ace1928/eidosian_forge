import threading
import torch._C._lazy
from torch.utils._pytree import tree_flatten, tree_unflatten
from .closure import add_step_closure, run_step_closures
def wait_device_ops(devices=None):
    """Waits for all the async operations on the given devices to complete.
    Args:
      devices (string..., optional): The devices whose async ops need to be waited
        for. If empty, all the local devices will be waited for.
    """
    if devices is None:
        devices = []
    torch._C._lazy._wait_device_ops(devices=devices)