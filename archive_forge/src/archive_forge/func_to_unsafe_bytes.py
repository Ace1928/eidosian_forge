from __future__ import (absolute_import, division, print_function)
import sys
import types
import warnings
from sys import intern as _sys_intern
from collections.abc import Mapping, Set
from ansible.module_utils.common.text.converters import to_bytes, to_text
from ansible.module_utils.common.collections import is_sequence
from ansible.utils.native_jinja import NativeJinjaText
def to_unsafe_bytes(*args, **kwargs):
    return wrap_var(to_bytes(*args, **kwargs))