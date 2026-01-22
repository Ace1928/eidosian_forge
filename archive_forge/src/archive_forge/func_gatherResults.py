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
def gatherResults(deferredList: Iterable[Deferred[_T]], consumeErrors: bool=False) -> Deferred[List[_T]]:
    """
    Returns, via a L{Deferred}, a list with the results of the given
    L{Deferred}s - in effect, a "join" of multiple deferred operations.

    The returned L{Deferred} will fire when I{all} of the provided L{Deferred}s
    have fired, or when any one of them has failed.

    This method can be cancelled by calling the C{cancel} method of the
    L{Deferred}, all the L{Deferred}s in the list will be cancelled.

    This differs from L{DeferredList} in that you don't need to parse
    the result for success/failure.

    @param consumeErrors: (keyword param) a flag, defaulting to False,
        indicating that failures in any of the given L{Deferred}s should not be
        propagated to errbacks added to the individual L{Deferred}s after this
        L{gatherResults} invocation.  Any such errors in the individual
        L{Deferred}s will be converted to a callback result of L{None}.  This
        is useful to prevent spurious 'Unhandled error in Deferred' messages
        from being logged.  This parameter is available since 11.1.0.
    """
    return DeferredList(deferredList, fireOnOneErrback=True, consumeErrors=consumeErrors).addCallback(_parseDeferredListResult)