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
def read_env_file(file_path: StrPath, *, encoding: str=None, case_sensitive: bool=False) -> Dict[str, Optional[str]]:
    try:
        from dotenv import dotenv_values
    except ImportError as e:
        raise ImportError('python-dotenv is not installed, run `pip install pydantic[dotenv]`') from e
    file_vars: Dict[str, Optional[str]] = dotenv_values(file_path, encoding=encoding or 'utf8')
    if not case_sensitive:
        return {k.lower(): v for k, v in file_vars.items()}
    else:
        return file_vars