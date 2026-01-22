import math
import os
import sys
import platform
import textwrap
import ctypes
import inspect
from ._utils import _import_dotted_name, classproperty
from ._utils import _functionalize_sync as _sync
from ._utils_internal import get_file_path, prepare_multiprocessing_environment, \
from typing import Any, Callable, Dict, Optional, Set, Tuple, Type, TYPE_CHECKING, Union, List
import builtins
from math import e , nan , inf , pi
from ._tensor import Tensor
from .storage import _StorageBase, TypedStorage, _LegacyStorage, UntypedStorage, _warn_typed_storage_removal
from .random import set_rng_state, get_rng_state, manual_seed, initial_seed, seed
from .serialization import save, load
from ._tensor_str import set_printoptions
from torch.amp import autocast
from ._compile import _disable_dynamo
from .functional import *  # noqa: F403
from torch import cuda as cuda
from torch import cpu as cpu
from torch import mps as mps
from torch import autograd as autograd
from torch.autograd import (
from torch import fft as fft
from torch import futures as futures
from torch import _awaits as _awaits
from torch import nested as nested
from torch import nn as nn
from torch.signal import windows as windows
from torch import optim as optim
import torch.optim._multi_tensor
from torch import multiprocessing as multiprocessing
from torch import sparse as sparse
from torch import special as special
import torch.utils.backcompat
from torch import jit as jit
from torch import linalg as linalg
from torch import hub as hub
from torch import random as random
from torch import distributions as distributions
from torch import testing as testing
from torch import backends as backends
import torch.utils.data
from torch import __config__ as __config__
from torch import __future__ as __future__
from torch import profiler as profiler
from torch import ao as ao
import torch.nn.quantizable
import torch.nn.quantized
import torch.nn.qat
import torch.nn.intrinsic
from . import _torch_docs, _tensor_docs, _storage_docs
from torch._ops import ops
from torch._classes import classes
import torch._library
from torch import quantization as quantization
from torch import quasirandom as quasirandom
from torch.multiprocessing._atfork import register_after_fork
from ._lobpcg import lobpcg as lobpcg
from torch.utils.dlpack import from_dlpack, to_dlpack
from . import masked
from ._linalg_utils import (  # type: ignore[misc]
from ._linalg_utils import _symeig as symeig  # type: ignore[misc]
from torch import export as export
from torch._higher_order_ops import cond
from . import return_types
from . import library
import torch.fx.experimental.sym_node
from torch import func as func
from torch.func import vmap
from . import _logging
def set_default_device(device):
    """Sets the default ``torch.Tensor`` to be allocated on ``device``.  This
    does not affect factory function calls which are called with an explicit
    ``device`` argument.  Factory calls will be performed as if they
    were passed ``device`` as an argument.

    To only temporarily change the default device instead of setting it
    globally, use ``with torch.device(device):`` instead.

    The default device is initially ``cpu``.  If you set the default tensor
    device to another device (e.g., ``cuda``) without a device index, tensors
    will be allocated on whatever the current device for the device type,
    even after :func:`torch.cuda.set_device` is called.

    .. warning::

        This function imposes a slight performance cost on every Python
        call to the torch API (not just factory functions).  If this
        is causing problems for you, please comment on
        https://github.com/pytorch/pytorch/issues/92701

    .. note::

        This doesn't affect functions that create tensors that share the same memory as the input, like:
        :func:`torch.from_numpy` and :func:`torch.frombuffer`

    Args:
        device (device or string): the device to set as default

    Example::

        >>> # xdoctest: +SKIP("requires cuda, changes global state")
        >>> torch.tensor([1.2, 3]).device
        device(type='cpu')
        >>> torch.set_default_device('cuda')  # current device is 0
        >>> torch.tensor([1.2, 3]).device
        device(type='cuda', index=0)
        >>> torch.set_default_device('cuda:1')
        >>> torch.tensor([1.2, 3]).device
        device(type='cuda', index=1)

    """
    global _GLOBAL_DEVICE_CONTEXT
    if _GLOBAL_DEVICE_CONTEXT is not None:
        _GLOBAL_DEVICE_CONTEXT.__exit__(None, None, None)
    if device is None:
        _GLOBAL_DEVICE_CONTEXT = None
        return
    from torch.utils._device import DeviceContext
    _GLOBAL_DEVICE_CONTEXT = DeviceContext(device)
    _GLOBAL_DEVICE_CONTEXT.__enter__()