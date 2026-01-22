import os
import re
import sys
import traceback
import ast
import importlib
from re import sub, findall
from types import CodeType
from functools import partial
from collections import OrderedDict, defaultdict
import kivy.lang.builder  # imported as absolute to avoid circular import
from kivy.logger import Logger
from kivy.cache import Cache
from kivy import require
from kivy.resources import resource_find
from kivy.utils import rgba
import kivy.metrics as Metrics
def create_missing(self, widget):
    cls = widget.__class__
    if cls in self.cache_marked:
        return
    self.cache_marked.append(cls)
    for name in self.properties:
        if hasattr(widget, name):
            continue
        value = self.properties[name].co_value
        if type(value) is CodeType:
            value = None
        widget.create_property(name, value, default_value=False)