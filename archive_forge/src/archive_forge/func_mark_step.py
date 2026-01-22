import threading
import torch._C._lazy
from torch.utils._pytree import tree_flatten, tree_unflatten
from .closure import add_step_closure, run_step_closures
def mark_step(device: str='', wait=False):
    """Triggers a mark step, which amounts to
    - collecting a group of 'live' lazy tensors to index into the compilation cache
      (lowering/compiling their IR graphs if not cached)
    - kicking off execution of the compiled function
    - (optionally, wait=True) waiting for cpu-side execution to complete (does not sync the accelerator)
    """
    torch._C._lazy._mark_step(device, [], wait=wait)
    run_step_closures()