import ast
import contextlib
import logging
import os
import re
from typing import ClassVar, Sequence
import panel as pn
from .core import OpenFile, get_filesystem_class, split_protocol
from .registry import known_implementations
def home_clicked(self, *_):
    self.protocol.value = self.init_protocol
    self.kwargs.value = self.init_kwargs
    self.url.value = self.init_url
    self.go_clicked()