import os
import pickle
import warnings
from functools import partial
from io import BytesIO
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Callable, Dict, Optional, OrderedDict, Sequence, Set, Tuple, Union
import torch
from lightning_utilities.core.apply_func import apply_to_collection
from torch import Tensor
from torch._C import _TensorMeta
from torch.nn import Parameter
from typing_extensions import override
from lightning_fabric.utilities.imports import (
from lightning_fabric.utilities.types import _PATH, _Stateful
@classmethod
def rebuild_tensor_v2(cls, storage: 'TypedStorage', storage_offset: int, size: tuple, stride: tuple, requires_grad: bool, backward_hooks: OrderedDict, metadata: Optional[Any]=None, *, archiveinfo: '_LazyLoadingUnpickler') -> '_NotYetLoadedTensor':
    rebuild_args = (storage_offset, size, stride, requires_grad, backward_hooks, metadata)
    metatensor = torch._utils._rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks, metadata)
    storageinfo = storage.archiveinfo
    return _NotYetLoadedTensor(metatensor, archiveinfo, storageinfo, rebuild_args)