from __future__ import annotations
import sys
import warnings
from io import StringIO
from typing import Mapping, Sequence, TypeVar
from unittest import TestResult
from twisted.python.filepath import FilePath
from twisted.trial._synctest import (
from twisted.trial.unittest import SynchronousTestCase
import warnings
import warnings
def test_setWarningRegistryChangeWhileIterating(self) -> None:
    """
        If the dictionary passed to L{_setWarningRegistryToNone} changes size
        partway through the process, C{_setWarningRegistryToNone} continues to
        set C{__warningregistry__} to L{None} on the rest of the values anyway.


        This might be caused by C{sys.modules} containing something that's not
        really a module and imports things on setattr.  py.test does this, as
        does L{twisted.python.deprecate.deprecatedModuleAttribute}.
        """
    d: dict[object, A | None] = {}

    class A:

        def __init__(self, key: object) -> None:
            self.__dict__['_key'] = key

        def __setattr__(self, value: object, item: object) -> None:
            d[self._key] = None
    key1 = object()
    key2 = object()
    d[key1] = A(key2)
    key3 = object()
    key4 = object()
    d[key3] = A(key4)
    _setWarningRegistryToNone(d)
    self.assertEqual({key1, key2, key3, key4}, set(d.keys()))