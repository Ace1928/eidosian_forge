import os
import functools
from enum import Enum
from pathlib import Path
from lazyops.types.models import BaseSettings, pre_root_validator, validator
from lazyops.imports._pydantic import BaseAppSettings, BaseModel
from lazyops.utils.system import is_in_kubernetes, get_host_name
from lazyops.utils.assets import create_get_assets_wrapper, create_import_assets_wrapper
from lazyops.libs.fastapi_utils import GlobalContext
from lazyops.libs.fastapi_utils.types.persistence import TemporaryData
from typing import List, Optional, Dict, Any, Callable, Union, Type, TYPE_CHECKING
def get_assets_func(module_name: str, asset_path: Optional[str]='assets') -> Callable[..., Union[Dict[str, Any], List[Any]]]:
    """
    Returns the get assets function
    """
    global _get_assets_wrappers
    if module_name not in _get_assets_wrappers:
        _get_assets_wrappers[module_name] = create_get_assets_wrapper(module_name, asset_path)
    return _get_assets_wrappers[module_name]