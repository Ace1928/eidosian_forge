import copy
import enum
import functools
import logging
import multiprocessing
import shlex
import sys
import threading
from oslo_config import cfg
from oslo_config import types
from oslo_utils import importutils
from oslo_privsep._i18n import _
from oslo_privsep import capabilities
from oslo_privsep import daemon
def entrypoint_with_timeout(self, timeout):
    """This is intended to be used as a decorator with timeout."""

    def wrap(func):

        @functools.wraps(func)
        def inner(*args, **kwargs):
            f = self._entrypoint(func)
            return f(*args, _wrap_timeout=timeout, **kwargs)
        setattr(inner, _ENTRYPOINT_ATTR, self)
        return inner
    return wrap