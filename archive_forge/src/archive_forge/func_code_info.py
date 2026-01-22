import sys
import types
import collections
import io
from opcode import *
from opcode import (
def code_info(x):
    """Formatted details of methods, functions, or code."""
    return _format_code_info(_get_code_object(x))