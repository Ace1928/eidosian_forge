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
def get_app_ingress(module_name: str, base_domain: str, app_env: AppEnv, app_host: Optional[str]=None, app_port: Optional[int]=None) -> str:
    """
    Retrieves the app ingress
    """
    if app_env.is_local:
        app_host = app_host or 'localhost'
        app_port = app_port or 8080
        return f'http://{app_host}:{app_port}'
    if app_env == AppEnv.DEVELOPMENT:
        return f'https://{module_name}-develop.{base_domain}'
    if app_env == AppEnv.STAGING:
        return f'https://{module_name}-staging.{base_domain}'
    if app_env == AppEnv.PRODUCTION:
        return f'https://{module_name}.{base_domain}'
    raise ValueError(f'Invalid app environment: {app_env}')