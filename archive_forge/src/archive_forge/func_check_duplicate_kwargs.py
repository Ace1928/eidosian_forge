from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Optional, Tuple, Type, TypeVar, Union, overload
from . import validator
from .config import Extra
from .errors import ConfigError
from .main import BaseModel, create_model
from .typing import get_all_type_hints
from .utils import to_camel
@validator(V_DUPLICATE_KWARGS, check_fields=False, allow_reuse=True)
def check_duplicate_kwargs(cls, v: Optional[List[str]]) -> None:
    if v is None:
        return
    plural = '' if len(v) == 1 else 's'
    keys = ', '.join(map(repr, v))
    raise TypeError(f'multiple values for argument{plural}: {keys}')