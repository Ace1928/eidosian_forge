import multiprocessing
import os
import threading
from multiprocessing.reduction import ForkingPickler
from multiprocessing.util import register_after_fork
from typing import Union
import torch
import torch.utils.hooks
from torch._namedtensor_internals import check_serializing_named_tensor
@classmethod
def from_weakref(cls, cdata):
    instance = cls.__new__(cls)
    instance.cdata = cdata
    instance._free_weak_ref = torch.Storage._free_weak_ref
    return instance