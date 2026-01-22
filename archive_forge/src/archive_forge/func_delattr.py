from contextlib import contextmanager
import os
import re
import sys
from typing import Any
from typing import final
from typing import Generator
from typing import List
from typing import Mapping
from typing import MutableMapping
from typing import Optional
from typing import overload
from typing import Tuple
from typing import TypeVar
from typing import Union
import warnings
from _pytest.fixtures import fixture
from _pytest.warning_types import PytestWarning
def delattr(self, target: Union[object, str], name: Union[str, Notset]=notset, raising: bool=True) -> None:
    """Delete attribute ``name`` from ``target``.

        If no ``name`` is specified and ``target`` is a string
        it will be interpreted as a dotted import path with the
        last part being the attribute name.

        Raises AttributeError it the attribute does not exist, unless
        ``raising`` is set to False.
        """
    __tracebackhide__ = True
    import inspect
    if isinstance(name, Notset):
        if not isinstance(target, str):
            raise TypeError('use delattr(target, name) or delattr(target) with target being a dotted import string')
        name, target = derive_importpath(target, raising)
    if not hasattr(target, name):
        if raising:
            raise AttributeError(name)
    else:
        oldval = getattr(target, name, notset)
        if inspect.isclass(target):
            oldval = target.__dict__.get(name, notset)
        self._setattr.append((target, name, oldval))
        delattr(target, name)