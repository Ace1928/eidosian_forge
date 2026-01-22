import inspect
import logging
from typing import Optional, Union
from ray.util import log_once
from ray.util.annotations import _mark_annotated
def patched_init(*args, **kwargs):
    if log_once(old or obj.__name__):
        deprecation_warning(old=old or obj.__name__, new=new, help=help, error=error)
    return obj_init(*args, **kwargs)