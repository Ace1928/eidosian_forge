from __future__ import annotations
import os
import re
from functools import partial
from dask.core import get_dependencies, ishashable, istask
from dask.utils import apply, funcname, import_required, key_split
def has_sub_tasks(task):
    """Returns True if the task has sub tasks"""
    if istask(task):
        return True
    elif isinstance(task, list):
        return any((has_sub_tasks(i) for i in task))
    else:
        return False