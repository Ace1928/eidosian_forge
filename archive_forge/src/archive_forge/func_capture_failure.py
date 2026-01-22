import collections.abc
import contextlib
import datetime
import functools
import inspect
import io
import os
import re
import socket
import sys
import threading
import types
import enum
from oslo_serialization import jsonutils
from oslo_serialization import msgpackutils
from oslo_utils import encodeutils
from oslo_utils import importutils
from oslo_utils import netutils
from oslo_utils import reflection
from taskflow.types import failure
@contextlib.contextmanager
def capture_failure():
    """Captures the occurring exception and provides a failure object back.

    This will save the current exception information and yield back a
    failure object for the caller to use (it will raise a runtime error if
    no active exception is being handled).

    This is useful since in some cases the exception context can be cleared,
    resulting in None being attempted to be saved after an exception handler is
    run. This can happen when eventlet switches greenthreads or when running an
    exception handler, code raises and catches an exception. In both
    cases the exception context will be cleared.

    To work around this, we save the exception state, yield a failure and
    then run other code.

    For example::

        >>> from taskflow.utils import misc
        >>>
        >>> def cleanup():
        ...     pass
        ...
        >>>
        >>> def save_failure(f):
        ...     print("Saving %s" % f)
        ...
        >>>
        >>> try:
        ...     raise IOError("Broken")
        ... except Exception:
        ...     with misc.capture_failure() as fail:
        ...         print("Activating cleanup")
        ...         cleanup()
        ...         save_failure(fail)
        ...
        Activating cleanup
        Saving Failure: IOError: Broken

    """
    exc_info = sys.exc_info()
    if not any(exc_info):
        raise RuntimeError('No active exception is being handled')
    else:
        yield failure.Failure(exc_info=exc_info)