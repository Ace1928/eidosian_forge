from __future__ import annotations
import inspect
import warnings
from collections import defaultdict
from functools import lru_cache
from typing import Callable
def inner_doc(cls):
    functions = list(fns)
    if hasattr(cls, 'EVENTS'):
        functions += cls.EVENTS
    if inherit:
        classes_inherit_documentation[cls] = None
    documentation_group = _documentation_group
    if _documentation_group is None:
        try:
            modname = inspect.getmodule(cls).__name__
            if modname.startswith('gradio.') or modname.startswith('gradio_client.'):
                documentation_group = _get_module_documentation_group(modname)
            else:
                pass
        except Exception as exc:
            warnings.warn(f'Could not get documentation group for {cls}: {exc}')
    classes_to_document[documentation_group].append((cls, functions))
    return cls