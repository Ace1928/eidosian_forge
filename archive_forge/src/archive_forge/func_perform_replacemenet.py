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
def perform_replacemenet(self, v):
    """
        Return v with any applicable regex replacements applied
        """
    if isinstance(v, str):
        for repl_args in self.regex_replacements:
            if repl_args:
                v = re.sub(repl_args[0], repl_args[1], v)
    return v