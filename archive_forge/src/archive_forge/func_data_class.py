import base64
import numbers
import textwrap
import uuid
from importlib import import_module
import copy
import io
import re
import sys
import warnings
from _plotly_utils.optional_imports import get_module
@property
def data_class(self):
    if self._data_class is None:
        module = import_module(self.module_str)
        self._data_class = getattr(module, self.data_class_str)
    return self._data_class