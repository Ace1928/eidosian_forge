from __future__ import annotations as _annotations
import operator
import sys
import types
import typing
import warnings
from copy import copy, deepcopy
from typing import Any, ClassVar
import pydantic_core
import typing_extensions
from pydantic_core import PydanticUndefined
from ._internal import (
from ._migration import getattr_migration
from .annotated_handlers import GetCoreSchemaHandler, GetJsonSchemaHandler
from .config import ConfigDict
from .errors import PydanticUndefinedAnnotation, PydanticUserError
from .json_schema import DEFAULT_REF_TEMPLATE, GenerateJsonSchema, JsonSchemaMode, JsonSchemaValue, model_json_schema
from .warnings import PydanticDeprecatedSince20
@classmethod
def model_construct(cls: type[Model], _fields_set: set[str] | None=None, **values: Any) -> Model:
    """Creates a new instance of the `Model` class with validated data.

        Creates a new model setting `__dict__` and `__pydantic_fields_set__` from trusted or pre-validated data.
        Default values are respected, but no other validation is performed.
        Behaves as if `Config.extra = 'allow'` was set since it adds all passed values

        Args:
            _fields_set: The set of field names accepted for the Model instance.
            values: Trusted or pre-validated data dictionary.

        Returns:
            A new instance of the `Model` class with validated data.
        """
    m = cls.__new__(cls)
    fields_values: dict[str, Any] = {}
    fields_set = set()
    for name, field in cls.model_fields.items():
        if field.alias and field.alias in values:
            fields_values[name] = values.pop(field.alias)
            fields_set.add(name)
        elif name in values:
            fields_values[name] = values.pop(name)
            fields_set.add(name)
        elif not field.is_required():
            fields_values[name] = field.get_default(call_default_factory=True)
    if _fields_set is None:
        _fields_set = fields_set
    _extra: dict[str, Any] | None = None
    if cls.model_config.get('extra') == 'allow':
        _extra = {}
        for k, v in values.items():
            _extra[k] = v
    else:
        fields_values.update(values)
    _object_setattr(m, '__dict__', fields_values)
    _object_setattr(m, '__pydantic_fields_set__', _fields_set)
    if not cls.__pydantic_root_model__:
        _object_setattr(m, '__pydantic_extra__', _extra)
    if cls.__pydantic_post_init__:
        m.model_post_init(None)
        if hasattr(m, '__pydantic_private__') and m.__pydantic_private__ is not None:
            for k, v in values.items():
                if k in m.__private_attributes__:
                    m.__pydantic_private__[k] = v
    elif not cls.__pydantic_root_model__:
        _object_setattr(m, '__pydantic_private__', None)
    return m