import abc
import contextlib
import os
import sys
import warnings
import numba.core.config
import numpy as np
from collections import defaultdict
from functools import wraps
from abc import abstractmethod
@contextlib.contextmanager
def catch_warnings(self, filename=None, lineno=None):
    """
        Store warnings and optionally fix their filename and lineno.
        """
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter('always', self._category)
        yield
    for w in wlist:
        msg = str(w.message)
        if issubclass(w.category, self._category):
            filename = filename or w.filename
            lineno = lineno or w.lineno
            self._warnings[filename, lineno, w.category].add(msg)
        else:
            warnings.warn_explicit(msg, w.category, w.filename, w.lineno)