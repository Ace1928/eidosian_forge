import multiprocessing
import os
import threading
from multiprocessing.reduction import ForkingPickler
from multiprocessing.util import register_after_fork
from typing import Union
import torch
import torch.utils.hooks
from torch._namedtensor_internals import check_serializing_named_tensor
def init_reductions():
    ForkingPickler.register(torch.cuda.Event, reduce_event)
    for t in torch._storage_classes:
        if t.__name__ == 'UntypedStorage':
            ForkingPickler.register(t, reduce_storage)
        else:
            ForkingPickler.register(t, reduce_typed_storage_child)
    ForkingPickler.register(torch.storage.TypedStorage, reduce_typed_storage)
    for t in torch._tensor_classes:
        ForkingPickler.register(t, reduce_tensor)
    ForkingPickler.register(torch.Tensor, reduce_tensor)
    ForkingPickler.register(torch.nn.parameter.Parameter, reduce_tensor)