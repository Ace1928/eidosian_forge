from __future__ import annotations
import contextvars
import functools
import gc
import re
import traceback
import types
import unittest as pyunit
import warnings
import weakref
from asyncio import (
from typing import (
from hamcrest import assert_that, empty, equal_to
from hypothesis import given
from hypothesis.strategies import integers
from typing_extensions import assert_type
from twisted.internet import defer, reactor
from twisted.internet.defer import (
from twisted.internet.task import Clock
from twisted.python import log
from twisted.python.compat import _PYPY
from twisted.python.failure import Failure
from twisted.trial import unittest
def testDeferredListFireOnOneError(self) -> None:
    defr1: Deferred[str] = Deferred()
    defr2: Deferred[str] = Deferred()
    defr3: Deferred[str] = Deferred()
    dl = DeferredList([defr1, defr2, defr3], fireOnOneErrback=True)
    result: List[Failure] = []
    dl.addErrback(result.append)

    def catch(err: Failure) -> None:
        return None
    defr2.addErrback(catch)
    defr1.callback('1')
    self.assertEqual(result, [])
    defr2.errback(GenericError('from def2'))
    self.assertEqual(len(result), 1)
    aFailure = result[0]
    self.assertTrue(issubclass(aFailure.type, defer.FirstError), "issubclass(aFailure.type, defer.FirstError) failed: failure's type is %r" % (aFailure.type,))
    firstError = aFailure.value
    self.assertEqual(firstError.subFailure.type, GenericError)
    self.assertEqual(firstError.subFailure.value.args, ('from def2',))
    self.assertEqual(firstError.index, 1)