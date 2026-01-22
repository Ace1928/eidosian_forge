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
def get_app_default_file(configs_path: Path, name: Optional[str]=None, required: Optional[bool]=False, suffix: Optional[str]=None, module_name: Optional[str]=None) -> Optional[Path]:
    """
    Retrieves the app environment file

    Only valid for local/dev environments
    """
    app_env = get_app_env(module_name)
    defaults_path = configs_path.joinpath('defaults')
    suffix = suffix or 'json'
    if name is not None:
        if defaults_path.joinpath(f'{name}-{app_env.name}.{suffix}').exists():
            return defaults_path.joinpath(f'{name}-{app_env.name}.{suffix}')
        if defaults_path.joinpath(f'{name}.{suffix}').exists():
            return defaults_path.joinpath(f'{name}.{suffix}')
        if required:
            raise ValueError(f'Invalid app environment file: {name}')
    env_path = defaults_path.joinpath(f'{app_env.name}.{suffix}')
    if env_path.exists():
        return env_path
    default_path = defaults_path.joinpath(f'default.{suffix}')
    return default_path if default_path.exists() else None