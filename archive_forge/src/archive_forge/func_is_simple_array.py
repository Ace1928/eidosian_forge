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
def is_simple_array(v):
    """
    Return whether a value is considered to be an simple array
    """
    return isinstance(v, (list, tuple))