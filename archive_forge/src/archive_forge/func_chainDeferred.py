from __future__ import annotations
import inspect
import traceback
import warnings
from abc import ABC, abstractmethod
from asyncio import AbstractEventLoop, Future, iscoroutine
from contextvars import Context as _Context, copy_context as _copy_context
from enum import Enum
from functools import wraps
from sys import exc_info, implementation
from types import CoroutineType, GeneratorType, MappingProxyType, TracebackType
from typing import (
import attr
from incremental import Version
from typing_extensions import Concatenate, Literal, ParamSpec, Self
from twisted.internet.interfaces import IDelayedCall, IReactorTime
from twisted.logger import Logger
from twisted.python import lockfile
from twisted.python.compat import _PYPY, cmp, comparable
from twisted.python.deprecate import deprecated, warnAboutFunction
from twisted.python.failure import Failure, _extraneous
def chainDeferred(self, d: 'Deferred[_SelfResultT]') -> 'Deferred[None]':
    """
        Chain another L{Deferred} to this L{Deferred}.

        This method adds callbacks to this L{Deferred} to call C{d}'s callback
        or errback, as appropriate. It is merely a shorthand way of performing
        the following::

            d1.addCallbacks(d2.callback, d2.errback)

        When you chain a deferred C{d2} to another deferred C{d1} with
        C{d1.chainDeferred(d2)}, you are making C{d2} participate in the
        callback chain of C{d1}.
        Thus any event that fires C{d1} will also fire C{d2}.
        However, the converse is B{not} true; if C{d2} is fired, C{d1} will not
        be affected.

        Note that unlike the case where chaining is caused by a L{Deferred}
        being returned from a callback, it is possible to cause the call
        stack size limit to be exceeded by chaining many L{Deferred}s
        together with C{chainDeferred}.

        @return: C{self}.
        """
    d._chainedTo = self
    return self.addCallbacks(d.callback, d.errback)