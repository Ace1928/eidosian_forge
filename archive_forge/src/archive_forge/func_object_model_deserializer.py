import uuid
import json
import typing
import codecs
import hashlib
import datetime
import contextlib
import dataclasses
from enum import Enum
from .lazy import lazy_import, get_obj_class_name
def object_model_deserializer(obj: typing.Any) -> typing.Union['BaseModel', typing.Any]:
    """
    Hooks for the object deserializer for BaseModels
    """
    if not isinstance(obj, dict):
        return obj
    if any((key not in obj for key in ['__jsontype__', '__model__', '__data__'])):
        return object_deserializer(obj)
    model = lazy_import(obj['__model__'])
    return model(**obj['__data__'])