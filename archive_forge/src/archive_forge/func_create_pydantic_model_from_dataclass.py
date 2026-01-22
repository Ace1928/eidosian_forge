import copy
import dataclasses
import sys
from contextlib import contextmanager
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Dict, Generator, Optional, Type, TypeVar, Union, overload
from typing_extensions import dataclass_transform
from .class_validators import gather_all_validators
from .config import BaseConfig, ConfigDict, Extra, get_config
from .error_wrappers import ValidationError
from .errors import DataclassTypeError
from .fields import Field, FieldInfo, Required, Undefined
from .main import create_model, validate_model
from .utils import ClassAttribute
def create_pydantic_model_from_dataclass(dc_cls: Type['Dataclass'], config: Type[Any]=BaseConfig, dc_cls_doc: Optional[str]=None) -> Type['BaseModel']:
    field_definitions: Dict[str, Any] = {}
    for field in dataclasses.fields(dc_cls):
        default: Any = Undefined
        default_factory: Optional['NoArgAnyCallable'] = None
        field_info: FieldInfo
        if field.default is not dataclasses.MISSING:
            default = field.default
        elif field.default_factory is not dataclasses.MISSING:
            default_factory = field.default_factory
        else:
            default = Required
        if isinstance(default, FieldInfo):
            field_info = default
            dc_cls.__pydantic_has_field_info_default__ = True
        else:
            field_info = Field(default=default, default_factory=default_factory, **field.metadata)
        field_definitions[field.name] = (field.type, field_info)
    validators = gather_all_validators(dc_cls)
    model: Type['BaseModel'] = create_model(dc_cls.__name__, __config__=config, __module__=dc_cls.__module__, __validators__=validators, __cls_kwargs__={'__resolve_forward_refs__': False}, **field_definitions)
    model.__doc__ = dc_cls_doc if dc_cls_doc is not None else dc_cls.__doc__ or ''
    return model