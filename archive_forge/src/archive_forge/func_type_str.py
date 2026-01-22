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
def type_str(v):
    """
    Return a type string of the form module.name for the input value v
    """
    if not isinstance(v, type):
        v = type(v)
    return "'{module}.{name}'".format(module=v.__module__, name=v.__name__)