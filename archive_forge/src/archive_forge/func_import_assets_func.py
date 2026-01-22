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
def import_assets_func(module_name: str, asset_path: Optional[str]='assets') -> Callable[..., Union[Path, Any, List[Path], List[Any], Dict[str, Path], Dict[str, Any]]]:
    """
    Returns the import assets function
    """
    global _import_assets_wrappers
    if module_name not in _import_assets_wrappers:
        _import_assets_wrappers[module_name] = create_import_assets_wrapper(module_name, asset_path)
    return _import_assets_wrappers[module_name]