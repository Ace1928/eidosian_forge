from __future__ import annotations
from argparse import Namespace
import collections
import inspect
import typing
from typing import Any
from typing import Callable
from typing import Iterable
from typing import NoReturn
from typing import Optional
from typing import Tuple
from typing import TypeVar
from typing import Union
from . import mock
from . import requirements as _requirements
from .util import fail
from .. import util
def combinations_list(arg_iterable: Iterable[Tuple[Any, ...]], **kw):
    """As combination, but takes a single iterable"""
    return combinations(*arg_iterable, **kw)