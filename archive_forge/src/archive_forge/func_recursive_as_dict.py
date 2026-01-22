from __future__ import annotations
import datetime
import json
import os
import pathlib
import traceback
import types
from collections import OrderedDict, defaultdict
from enum import Enum
from hashlib import sha1
from importlib import import_module
from inspect import getfullargspec
from pathlib import Path
from uuid import UUID
def recursive_as_dict(obj):
    if isinstance(obj, (list, tuple)):
        return [recursive_as_dict(it) for it in obj]
    if isinstance(obj, dict):
        return {kk: recursive_as_dict(vv) for kk, vv in obj.items()}
    if hasattr(obj, 'as_dict'):
        return obj.as_dict()
    if dataclasses is not None and dataclasses.is_dataclass(obj):
        d = dataclasses.asdict(obj)
        d.update({'@module': obj.__class__.__module__, '@class': obj.__class__.__name__})
        return d
    return obj