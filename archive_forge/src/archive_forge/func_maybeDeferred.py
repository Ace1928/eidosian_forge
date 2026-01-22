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
def maybeDeferred(f: Callable[_P, Union[Deferred[_T], Coroutine[Deferred[Any], Any, _T], _T]], *args: _P.args, **kwargs: _P.kwargs) -> 'Deferred[_T]':
    """
    Invoke a function that may or may not return a L{Deferred} or coroutine.

    Call the given function with the given arguments.  Then:

      - If the returned object is a L{Deferred}, return it.

      - If the returned object is a L{Failure}, wrap it with L{fail} and
        return it.

      - If the returned object is a L{types.CoroutineType}, wrap it with
        L{Deferred.fromCoroutine} and return it.

      - Otherwise, wrap it in L{succeed} and return it.

      - If an exception is raised, convert it to a L{Failure}, wrap it in
        L{fail}, and then return it.

    @param f: The callable to invoke
    @param args: The arguments to pass to C{f}
    @param kwargs: The keyword arguments to pass to C{f}

    @return: The result of the function call, wrapped in a L{Deferred} if
    necessary.
    """
    try:
        result = f(*args, **kwargs)
    except BaseException:
        return fail(Failure(captureVars=Deferred.debug))
    if isinstance(result, Deferred):
        return result
    elif isinstance(result, Failure):
        return fail(result)
    elif type(result) is CoroutineType:
        return Deferred.fromCoroutine(result)
    else:
        returned: _T = result
        return succeed(returned)