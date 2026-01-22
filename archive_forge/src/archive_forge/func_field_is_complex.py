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
def field_is_complex(self, field: ModelField) -> Tuple[bool, bool]:
    """
        Find out if a field is complex, and if so whether JSON errors should be ignored
        """
    if lenient_issubclass(field.annotation, JsonWrapper):
        return (False, False)
    if field.is_complex():
        allow_parse_failure = False
    elif is_union(get_origin(field.type_)) and field.sub_fields and any((f.is_complex() for f in field.sub_fields)):
        allow_parse_failure = True
    else:
        return (False, False)
    return (True, allow_parse_failure)