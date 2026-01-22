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
def precompile(self):
    for x in self.properties.values():
        x.precompile()
    for x in self.handlers:
        x.precompile()
    for x in self.children:
        x.precompile()
    if self.canvas_before:
        self.canvas_before.precompile()
    if self.canvas_root:
        self.canvas_root.precompile()
    if self.canvas_after:
        self.canvas_after.precompile()