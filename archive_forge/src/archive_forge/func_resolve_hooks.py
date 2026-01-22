from __future__ import annotations
import copy
import datetime as dt
import decimal
import inspect
import json
import typing
import uuid
import warnings
from abc import ABCMeta
from collections import OrderedDict, defaultdict
from collections.abc import Mapping
from functools import lru_cache
from marshmallow import base, class_registry, types
from marshmallow import fields as ma_fields
from marshmallow.decorators import (
from marshmallow.error_store import ErrorStore
from marshmallow.exceptions import StringNotCollectionError, ValidationError
from marshmallow.orderedset import OrderedSet
from marshmallow.utils import (
from marshmallow.warnings import RemovedInMarshmallow4Warning
def resolve_hooks(cls) -> dict[types.Tag, list[str]]:
    """Add in the decorated processors

        By doing this after constructing the class, we let standard inheritance
        do all the hard work.
        """
    mro = inspect.getmro(cls)
    hooks = defaultdict(list)
    for attr_name in dir(cls):
        for parent in mro:
            try:
                attr = parent.__dict__[attr_name]
            except KeyError:
                continue
            else:
                break
        else:
            continue
        try:
            hook_config = attr.__marshmallow_hook__
        except AttributeError:
            pass
        else:
            for key in hook_config.keys():
                hooks[key].append(attr_name)
    return hooks