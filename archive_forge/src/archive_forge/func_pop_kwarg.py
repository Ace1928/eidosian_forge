from __future__ import annotations
import inspect
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import List
from typing import Optional
from typing import overload
from typing import Type
from typing import Union
from . import base
from . import url as _url
from .interfaces import DBAPIConnection
from .mock import create_mock_engine
from .. import event
from .. import exc
from .. import util
from ..pool import _AdhocProxiedConnection
from ..pool import ConnectionPoolEntry
from ..sql import compiler
from ..util import immutabledict
def pop_kwarg(key: str, default: Optional[Any]=None) -> Any:
    value = kwargs.pop(key, default)
    if key in dialect_cls.engine_config_types:
        value = dialect_cls.engine_config_types[key](value)
    return value