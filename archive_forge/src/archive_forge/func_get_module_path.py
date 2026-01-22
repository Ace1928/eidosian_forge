import typing
import functools
import pathlib
import importlib.util
from lazyops.utils.lazy import lazy_import
from typing import Optional, Dict, List, Union, Any, Callable, Type, TYPE_CHECKING
@functools.lru_cache()
def get_module_path(module_name: str, **kwargs) -> pathlib.Path:
    """
    Get the path to the module.

    args:
        module_name: name of the module to import from (e.g. 'lazyops')
    
    Use it like this:

    >>> get_module_path('lazyops')
    """
    module_spec = importlib.util.find_spec(module_name)
    if not module_spec:
        raise ValueError(f'Module {module_name} not found')
    for path in module_spec.submodule_search_locations:
        module_path = pathlib.Path(path)
        if module_path.exists():
            return module_path
    raise ValueError(f'Module {module_name} cant be found in the path')