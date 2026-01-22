import collections
import contextlib
import re
import torch
from typing import Callable, Dict, Optional, Tuple, Type, Union
def collate_numpy_scalar_fn(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]]=None):
    return torch.as_tensor(batch)