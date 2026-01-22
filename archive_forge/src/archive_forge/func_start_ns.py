import sys
import re
import warnings
import io
import collections
import collections.abc
import contextlib
import weakref
from . import ElementPath
fromstring = XML
def start_ns(self, prefix, uri):
    if self._ignored_depth:
        return
    if self._data:
        self._flush()
    self._ns_stack[-1].append((uri, prefix))