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
def in_values(self, e):
    """
        Return whether a value matches one of the enumeration options
        """
    is_str = isinstance(e, str)
    for v, regex in zip(self.values, self.val_regexs):
        if is_str and regex:
            in_values = fullmatch(regex, e) is not None
        else:
            in_values = e == v
        if in_values:
            return True
    return False