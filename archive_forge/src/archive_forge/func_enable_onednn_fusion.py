import warnings
from contextlib import contextmanager
from typing import Any, Iterator
import torch._C
from torch._jit_internal import (
from torch.jit._async import fork, wait
from torch.jit._await import _awaitable, _awaitable_nowait, _awaitable_wait
from torch.jit._decomposition_utils import _register_decomposition
from torch.jit._freeze import freeze, optimize_for_inference, run_frozen_optimizations
from torch.jit._fuser import (
from torch.jit._ir_utils import _InsertPoint
from torch.jit._script import (
from torch.jit._serialization import (
from torch.jit._trace import (
from torch.utils import set_module
def enable_onednn_fusion(enabled: bool):
    """Enable or disables onednn JIT fusion based on the parameter `enabled`."""
    torch._C._jit_set_llga_enabled(enabled)