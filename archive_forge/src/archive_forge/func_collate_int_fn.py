import collections
import contextlib
import re
import torch
from typing import Callable, Dict, Optional, Tuple, Type, Union
def collate_int_fn(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]]=None):
    return torch.tensor(batch)