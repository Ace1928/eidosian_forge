import ast
import contextlib
import logging
import os
import re
from typing import ClassVar, Sequence
import panel as pn
from .core import OpenFile, get_filesystem_class, split_protocol
from .registry import known_implementations
@contextlib.contextmanager
def ignore_events(self):
    """Temporarily turn off events processing in this instance

        (does not propagate to children)
        """
    self._ignoring_events = True
    try:
        yield
    finally:
        self._ignoring_events = False