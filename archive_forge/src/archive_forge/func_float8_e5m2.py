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
def float8_e5m2(self):
    """Casts this storage to float8_e5m2 type"""
    _warn_typed_storage_removal()
    return self._to(torch.float8_e5m2)