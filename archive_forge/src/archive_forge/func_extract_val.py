from typing import Optional
import torch.fx
from torch.fx import Node
from torch.fx._compatibility import compatibility
from torch._subclasses.fake_tensor import FakeTensorMode, FakeTensor
from torch.fx.experimental.proxy_tensor import py_sym_types, snapshot_fake
from torch.fx.node import map_aggregate
def extract_val(obj):
    if isinstance(obj, FakeTensor):
        return snapshot_fake(obj)
    elif isinstance(obj, torch.Tensor):
        return snapshot_fake(self._mode.from_tensor(obj, static_shapes=True))
    elif isinstance(obj, py_sym_types):
        return obj
    else:
        return None