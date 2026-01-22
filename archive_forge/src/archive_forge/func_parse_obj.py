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
@typing_extensions.deprecated('The `parse_obj` method is deprecated; use `model_validate` instead.', category=None)
def parse_obj(cls: type[Model], obj: Any) -> Model:
    warnings.warn('The `parse_obj` method is deprecated; use `model_validate` instead.', category=PydanticDeprecatedSince20)
    return cls.model_validate(obj)