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
def object_model_serializer(obj: typing.Union['BaseModel', typing.Any]) -> typing.Any:
    """
    Hooks for the object serializer for BaseModels
    """
    if not hasattr(obj, 'Config') and (not hasattr(obj, 'dict')):
        return object_serializer(obj)
    return {'__jsontype__': 'model', '__model__': get_obj_class_name(obj), '__data__': obj.dict()}