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
@wraps(init)
def handle_extra_init(self: 'Dataclass', *args: Any, **kwargs: Any) -> None:
    if config.extra == Extra.ignore:
        init(self, *args, **{k: v for k, v in kwargs.items() if k in self.__dataclass_fields__})
    elif config.extra == Extra.allow:
        for k, v in kwargs.items():
            self.__dict__.setdefault(k, v)
        init(self, *args, **{k: v for k, v in kwargs.items() if k in self.__dataclass_fields__})
    else:
        init(self, *args, **kwargs)