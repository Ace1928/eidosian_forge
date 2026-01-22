from __future__ import annotations
import inspect
from typing import TYPE_CHECKING, Any, Type, Union, Generic, TypeVar, Callable, cast
from datetime import date, datetime
from typing_extensions import (
import pydantic
import pydantic.generics
from pydantic.fields import FieldInfo
from ._types import (
from ._utils import (
from ._compat import (
from ._constants import RAW_RESPONSE_HEADER
def construct_type(*, value: object, type_: type) -> object:
    """Loose coercion to the expected type with construction of nested values.

    If the given value does not match the expected type then it is returned as-is.
    """
    if is_annotated_type(type_):
        meta = get_args(type_)[1:]
        type_ = extract_type_arg(type_, 0)
    else:
        meta = tuple()
    origin = get_origin(type_) or type_
    args = get_args(type_)
    if is_union(origin):
        try:
            return validate_type(type_=cast('type[object]', type_), value=value)
        except Exception:
            pass
        discriminator = _build_discriminated_union_meta(union=type_, meta_annotations=meta)
        if discriminator and is_mapping(value):
            variant_value = value.get(discriminator.field_alias_from or discriminator.field_name)
            if variant_value and isinstance(variant_value, str):
                variant_type = discriminator.mapping.get(variant_value)
                if variant_type:
                    return construct_type(type_=variant_type, value=value)
        for variant in args:
            try:
                return construct_type(value=value, type_=variant)
            except Exception:
                continue
        raise RuntimeError(f'Could not convert data into a valid instance of {type_}')
    if origin == dict:
        if not is_mapping(value):
            return value
        _, items_type = get_args(type_)
        return {key: construct_type(value=item, type_=items_type) for key, item in value.items()}
    if not is_literal_type(type_) and (issubclass(origin, BaseModel) or issubclass(origin, GenericModel)):
        if is_list(value):
            return [cast(Any, type_).construct(**entry) if is_mapping(entry) else entry for entry in value]
        if is_mapping(value):
            if issubclass(type_, BaseModel):
                return type_.construct(**value)
            return cast(Any, type_).construct(**value)
    if origin == list:
        if not is_list(value):
            return value
        inner_type = args[0]
        return [construct_type(value=entry, type_=inner_type) for entry in value]
    if origin == float:
        if isinstance(value, int):
            coerced = float(value)
            if coerced != value:
                return value
            return coerced
        return value
    if type_ == datetime:
        try:
            return parse_datetime(value)
        except Exception:
            return value
    if type_ == date:
        try:
            return parse_date(value)
        except Exception:
            return value
    return value