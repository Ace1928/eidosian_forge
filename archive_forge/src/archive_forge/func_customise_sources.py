import os
import warnings
from pathlib import Path
from typing import AbstractSet, Any, Callable, ClassVar, Dict, List, Mapping, Optional, Tuple, Type, Union
from .config import BaseConfig, Extra
from .fields import ModelField
from .main import BaseModel
from .types import JsonWrapper
from .typing import StrPath, display_as_type, get_origin, is_union
from .utils import deep_update, lenient_issubclass, path_type, sequence_like
@classmethod
def customise_sources(cls, init_settings: SettingsSourceCallable, env_settings: SettingsSourceCallable, file_secret_settings: SettingsSourceCallable) -> Tuple[SettingsSourceCallable, ...]:
    return (init_settings, env_settings, file_secret_settings)