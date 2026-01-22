from typing import List, Tuple, Union, Dict, Any, Set, Mapping
import collections
from dataclasses import dataclass
import torch
import torch.fx
from torch.fx.node import _get_qualified_name
from torch.fx._compatibility import compatibility
@compatibility(is_backward_compatible=False)
def get_acc_ops_name(k):
    if isinstance(k, str):
        return k
    elif k.__module__ and 'acc_ops' in k.__module__:
        return f'acc_ops.{k.__name__}'
    else:
        module = k.__module__.replace('torch._ops', 'torch.ops')
        return f'{(module if module else '')}.{k.__name__}'