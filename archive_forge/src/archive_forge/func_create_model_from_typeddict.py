import sys
from typing import TYPE_CHECKING, Any, Dict, FrozenSet, NamedTuple, Type
from .fields import Required
from .main import BaseModel, create_model
from .typing import is_typeddict, is_typeddict_special
def create_model_from_typeddict(typeddict_cls: Type['TypedDict'], **kwargs: Any) -> Type['BaseModel']:
    """
    Create a `BaseModel` based on the fields of a `TypedDict`.
    Since `typing.TypedDict` in Python 3.8 does not store runtime information about optional keys,
    we raise an error if this happens (see https://bugs.python.org/issue38834).
    """
    field_definitions: Dict[str, Any]
    if not hasattr(typeddict_cls, '__required_keys__'):
        raise TypeError('You should use `typing_extensions.TypedDict` instead of `typing.TypedDict` with Python < 3.9.2. Without it, there is no way to differentiate required and optional fields when subclassed.')
    if is_legacy_typeddict(typeddict_cls) and any((is_typeddict_special(t) for t in typeddict_cls.__annotations__.values())):
        raise TypeError('You should use `typing_extensions.TypedDict` instead of `typing.TypedDict` with Python < 3.11. Without it, there is no way to reflect Required/NotRequired keys.')
    required_keys: FrozenSet[str] = typeddict_cls.__required_keys__
    field_definitions = {field_name: (field_type, Required if field_name in required_keys else None) for field_name, field_type in typeddict_cls.__annotations__.items()}
    return create_model(typeddict_cls.__name__, **kwargs, **field_definitions)