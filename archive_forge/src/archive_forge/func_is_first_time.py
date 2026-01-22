import io
import torch
from ._utils import _type, _cuda, _hpu
from torch.types import Storage
from typing import cast, Any, Dict as _Dict, Optional as _Optional, TypeVar, Type, Union
import copy
import collections
from functools import lru_cache
import warnings
import threading
import functools
def is_first_time():
    if not hasattr(_warn_typed_storage_removal, 'has_warned'):
        return True
    else:
        return not _warn_typed_storage_removal.__dict__['has_warned']