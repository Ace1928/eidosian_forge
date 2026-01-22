from __future__ import annotations
import os
import re
from functools import partial
from dask.core import get_dependencies, ishashable, istask
from dask.utils import apply, funcname, import_required, key_split
def task_label(task):
    """Label for a task on a dot graph.

    Examples
    --------
    >>> from operator import add
    >>> task_label((add, 1, 2))
    'add'
    >>> task_label((add, (add, 1, 2), 3))
    'add(...)'
    """
    func = task[0]
    if func is apply:
        func = task[1]
    if hasattr(func, 'funcs'):
        if len(func.funcs) > 1:
            return f'{funcname(func.funcs[0])}(...)'
        else:
            head = funcname(func.funcs[0])
    else:
        head = funcname(func)
    if any((has_sub_tasks(i) for i in task[1:])):
        return f'{head}(...)'
    else:
        return head