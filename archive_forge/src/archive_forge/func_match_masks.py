import fnmatch
import functools
import inspect
import os
import warnings
from io import IOBase
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from torch.utils.data._utils.serialization import DILL_AVAILABLE
def match_masks(name: str, masks: Union[str, List[str]]) -> bool:
    if not masks:
        return True
    if isinstance(masks, str):
        return fnmatch.fnmatch(name, masks)
    for mask in masks:
        if fnmatch.fnmatch(name, mask):
            return True
    return False