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
@validator('app_env', pre=True)
def validate_app_env(cls, value: Optional[Any]) -> Any:
    """
        Validates the app environment
        """
    if value is None:
        return get_app_env(cls.__module__.split('.')[0])
    return AppEnv.from_env(value) if isinstance(value, str) else value