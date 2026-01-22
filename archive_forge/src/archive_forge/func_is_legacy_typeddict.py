import sys
from typing import TYPE_CHECKING, Any, Dict, FrozenSet, NamedTuple, Type
from .fields import Required
from .main import BaseModel, create_model
from .typing import is_typeddict, is_typeddict_special
def is_legacy_typeddict(_: Any) -> Any:
    return False