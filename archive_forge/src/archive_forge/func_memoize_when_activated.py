from __future__ import division
from __future__ import print_function
import collections
import contextlib
import errno
import functools
import os
import socket
import stat
import sys
import threading
import warnings
from collections import namedtuple
from socket import AF_INET
from socket import SOCK_DGRAM
from socket import SOCK_STREAM
def memoize_when_activated(fun):
    """A memoize decorator which is disabled by default. It can be
    activated and deactivated on request.
    For efficiency reasons it can be used only against class methods
    accepting no arguments.

    >>> class Foo:
    ...     @memoize
    ...     def foo()
    ...         print(1)
    ...
    >>> f = Foo()
    >>> # deactivated (default)
    >>> foo()
    1
    >>> foo()
    1
    >>>
    >>> # activated
    >>> foo.cache_activate(self)
    >>> foo()
    1
    >>> foo()
    >>> foo()
    >>>
    """

    @functools.wraps(fun)
    def wrapper(self):
        try:
            ret = self._cache[fun]
        except AttributeError:
            try:
                return fun(self)
            except Exception as err:
                raise raise_from(err, None)
        except KeyError:
            try:
                ret = fun(self)
            except Exception as err:
                raise raise_from(err, None)
            try:
                self._cache[fun] = ret
            except AttributeError:
                pass
        return ret

    def cache_activate(proc):
        """Activate cache. Expects a Process instance. Cache will be
        stored as a "_cache" instance attribute.
        """
        proc._cache = {}

    def cache_deactivate(proc):
        """Deactivate and clear cache."""
        try:
            del proc._cache
        except AttributeError:
            pass
    wrapper.cache_activate = cache_activate
    wrapper.cache_deactivate = cache_deactivate
    return wrapper