import collections
import copy
import gc
import itertools
import logging
import os
import queue
import sys
import threading
import time
from multiprocessing import TimeoutError
from typing import Any, Callable, Dict, Hashable, Iterable, List, Optional, Tuple
import ray
from ray._private.usage import usage_lib
from ray.util import log_once
def ray_put_if_needed(obj: Any, registry: Optional[List[Tuple[Any, ray.ObjectRef]]]=None, registry_hashable: Optional[Dict[Hashable, ray.ObjectRef]]=None) -> ray.ObjectRef:
    """ray.put obj in object store if it's not an ObjRef and bigger than 100 bytes,
    with support for list and dict registries"""
    if isinstance(obj, ray.ObjectRef) or sys.getsizeof(obj) < 100:
        return obj
    ret = obj
    if registry_hashable is not None:
        try:
            ret = _put_in_dict_registry(obj, registry_hashable)
        except TypeError:
            if registry is not None:
                ret = _put_in_list_registry(obj, registry)
    elif registry is not None:
        ret = _put_in_list_registry(obj, registry)
    return ret