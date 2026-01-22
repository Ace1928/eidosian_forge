import itertools
import random
from typing import Tuple
from .. import language as tl
from ..language import core as lcore
from . import torch_wrapper
from .core import ExecutionContext
from .memory_map import MemoryMap
from .tl_lang import (TritonLangProxy, WrappedTensor, _primitive_to_tensor,
def attach_triton(module, proxy):
    method_list = [func for func in dir(TritonLangProxy) if func[0] != '_']
    for name in method_list:
        if hasattr(module, name):
            attr = getattr(module, name)
            tl_method_backup[name] = attr
            if callable(attr):
                setattr(module, name, get_proxy_method(proxy, name))
            else:
                setattr(module, name, getattr(proxy, name))