import dataclasses
import datetime
from collections import defaultdict, deque
from decimal import Decimal
from enum import Enum
from ipaddress import (
from pathlib import Path, PurePath
from re import Pattern
from types import GeneratorType
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from uuid import UUID
from fastapi.types import IncEx
from pydantic import BaseModel
from pydantic.color import Color
from pydantic.networks import AnyUrl, NameEmail
from pydantic.types import SecretBytes, SecretStr
from typing_extensions import Annotated, Doc  # type: ignore [attr-defined]
from ._compat import PYDANTIC_V2, Url, _model_dump
def jsonable_encoder(obj: Annotated[Any, Doc('\n            The input object to convert to JSON.\n            ')], include: Annotated[Optional[IncEx], Doc("\n            Pydantic's `include` parameter, passed to Pydantic models to set the\n            fields to include.\n            ")]=None, exclude: Annotated[Optional[IncEx], Doc("\n            Pydantic's `exclude` parameter, passed to Pydantic models to set the\n            fields to exclude.\n            ")]=None, by_alias: Annotated[bool, Doc("\n            Pydantic's `by_alias` parameter, passed to Pydantic models to define if\n            the output should use the alias names (when provided) or the Python\n            attribute names. In an API, if you set an alias, it's probably because you\n            want to use it in the result, so you probably want to leave this set to\n            `True`.\n            ")]=True, exclude_unset: Annotated[bool, Doc("\n            Pydantic's `exclude_unset` parameter, passed to Pydantic models to define\n            if it should exclude from the output the fields that were not explicitly\n            set (and that only had their default values).\n            ")]=False, exclude_defaults: Annotated[bool, Doc("\n            Pydantic's `exclude_defaults` parameter, passed to Pydantic models to define\n            if it should exclude from the output the fields that had the same default\n            value, even when they were explicitly set.\n            ")]=False, exclude_none: Annotated[bool, Doc("\n            Pydantic's `exclude_none` parameter, passed to Pydantic models to define\n            if it should exclude from the output any fields that have a `None` value.\n            ")]=False, custom_encoder: Annotated[Optional[Dict[Any, Callable[[Any], Any]]], Doc("\n            Pydantic's `custom_encoder` parameter, passed to Pydantic models to define\n            a custom encoder.\n            ")]=None, sqlalchemy_safe: Annotated[bool, Doc("\n            Exclude from the output any fields that start with the name `_sa`.\n\n            This is mainly a hack for compatibility with SQLAlchemy objects, they\n            store internal SQLAlchemy-specific state in attributes named with `_sa`,\n            and those objects can't (and shouldn't be) serialized to JSON.\n            ")]=True) -> Any:
    """
    Convert any object to something that can be encoded in JSON.

    This is used internally by FastAPI to make sure anything you return can be
    encoded as JSON before it is sent to the client.

    You can also use it yourself, for example to convert objects before saving them
    in a database that supports only JSON.

    Read more about it in the
    [FastAPI docs for JSON Compatible Encoder](https://fastapi.tiangolo.com/tutorial/encoder/).
    """
    custom_encoder = custom_encoder or {}
    if custom_encoder:
        if type(obj) in custom_encoder:
            return custom_encoder[type(obj)](obj)
        else:
            for encoder_type, encoder_instance in custom_encoder.items():
                if isinstance(obj, encoder_type):
                    return encoder_instance(obj)
    if include is not None and (not isinstance(include, (set, dict))):
        include = set(include)
    if exclude is not None and (not isinstance(exclude, (set, dict))):
        exclude = set(exclude)
    if isinstance(obj, BaseModel):
        encoders: Dict[Any, Any] = {}
        if not PYDANTIC_V2:
            encoders = getattr(obj.__config__, 'json_encoders', {})
            if custom_encoder:
                encoders.update(custom_encoder)
        obj_dict = _model_dump(obj, mode='json', include=include, exclude=exclude, by_alias=by_alias, exclude_unset=exclude_unset, exclude_none=exclude_none, exclude_defaults=exclude_defaults)
        if '__root__' in obj_dict:
            obj_dict = obj_dict['__root__']
        return jsonable_encoder(obj_dict, exclude_none=exclude_none, exclude_defaults=exclude_defaults, custom_encoder=encoders, sqlalchemy_safe=sqlalchemy_safe)
    if dataclasses.is_dataclass(obj):
        obj_dict = dataclasses.asdict(obj)
        return jsonable_encoder(obj_dict, include=include, exclude=exclude, by_alias=by_alias, exclude_unset=exclude_unset, exclude_defaults=exclude_defaults, exclude_none=exclude_none, custom_encoder=custom_encoder, sqlalchemy_safe=sqlalchemy_safe)
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, PurePath):
        return str(obj)
    if isinstance(obj, (str, int, float, type(None))):
        return obj
    if isinstance(obj, dict):
        encoded_dict = {}
        allowed_keys = set(obj.keys())
        if include is not None:
            allowed_keys &= set(include)
        if exclude is not None:
            allowed_keys -= set(exclude)
        for key, value in obj.items():
            if (not sqlalchemy_safe or not isinstance(key, str) or (not key.startswith('_sa'))) and (value is not None or not exclude_none) and (key in allowed_keys):
                encoded_key = jsonable_encoder(key, by_alias=by_alias, exclude_unset=exclude_unset, exclude_none=exclude_none, custom_encoder=custom_encoder, sqlalchemy_safe=sqlalchemy_safe)
                encoded_value = jsonable_encoder(value, by_alias=by_alias, exclude_unset=exclude_unset, exclude_none=exclude_none, custom_encoder=custom_encoder, sqlalchemy_safe=sqlalchemy_safe)
                encoded_dict[encoded_key] = encoded_value
        return encoded_dict
    if isinstance(obj, (list, set, frozenset, GeneratorType, tuple, deque)):
        encoded_list = []
        for item in obj:
            encoded_list.append(jsonable_encoder(item, include=include, exclude=exclude, by_alias=by_alias, exclude_unset=exclude_unset, exclude_defaults=exclude_defaults, exclude_none=exclude_none, custom_encoder=custom_encoder, sqlalchemy_safe=sqlalchemy_safe))
        return encoded_list
    if type(obj) in ENCODERS_BY_TYPE:
        return ENCODERS_BY_TYPE[type(obj)](obj)
    for encoder, classes_tuple in encoders_by_class_tuples.items():
        if isinstance(obj, classes_tuple):
            return encoder(obj)
    try:
        data = dict(obj)
    except Exception as e:
        errors: List[Exception] = []
        errors.append(e)
        try:
            data = vars(obj)
        except Exception as e:
            errors.append(e)
            raise ValueError(errors) from e
    return jsonable_encoder(data, include=include, exclude=exclude, by_alias=by_alias, exclude_unset=exclude_unset, exclude_defaults=exclude_defaults, exclude_none=exclude_none, custom_encoder=custom_encoder, sqlalchemy_safe=sqlalchemy_safe)