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
def script_if_tracing(fn):
    """
    Compiles ``fn`` when it is first called during tracing.

    ``torch.jit.script`` has a non-negligible start up time when it is first called due to
    lazy-initializations of many compiler builtins. Therefore you should not use
    it in library code. However, you may want to have parts of your library work
    in tracing even if they use control flow. In these cases, you should use
    ``@torch.jit.script_if_tracing`` to substitute for
    ``torch.jit.script``.

    Args:
        fn: A function to compile.

    Returns:
        If called during tracing, a :class:`ScriptFunction` created by `torch.jit.script` is returned.
        Otherwise, the original function `fn` is returned.
    """
    return _script_if_tracing(fn)