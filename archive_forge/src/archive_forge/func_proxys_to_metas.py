import torch
import torch.fx
import warnings
import functools
import builtins
from typing import Any, Callable, Dict, Optional, Union
def proxys_to_metas(v):
    if isinstance(v, MetaDeviceAttribute):
        return 'meta'
    if isinstance(v, torch.fx.Proxy):
        assert isinstance(v, MetaProxy), f'Expected MetaProxy but got {type(v)}'
        assert hasattr(v, '_tensor_meta'), 'MetaProxy does not have an associated meta'
        return v._tensor_meta
    return v