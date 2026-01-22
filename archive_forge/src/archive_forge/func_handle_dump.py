import abc
import dataclasses
import functools
import inspect
import sys
from dataclasses import Field, fields
from typing import Any, Callable, Dict, Optional, Tuple, Union, Type, get_type_hints
from enum import Enum
from marshmallow.exceptions import ValidationError  # type: ignore
from dataclasses_json.utils import CatchAllVar
@staticmethod
def handle_dump(obj) -> Dict[Any, Any]:
    catch_all_field = _CatchAllUndefinedParameters._get_catch_all_field(cls=obj)
    return getattr(obj, catch_all_field.name)