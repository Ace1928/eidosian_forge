import ast
import contextlib
import logging
import os
import re
from typing import ClassVar, Sequence
import panel as pn
from .core import OpenFile, get_filesystem_class, split_protocol
from .registry import known_implementations
def protocol_changed(self, *_):
    self._fs = None
    self.main.options = []
    self.url.value = ''