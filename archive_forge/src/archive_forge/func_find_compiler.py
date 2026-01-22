from __future__ import annotations
from typing import Callable, Optional
from collections import OrderedDict
import os
import re
import subprocess
from .util import (
@classmethod
def find_compiler(cls, preferred_vendor=None):
    """ Identify a suitable C/fortran/other compiler. """
    candidates = list(cls.compiler_dict.keys())
    if preferred_vendor:
        if preferred_vendor in candidates:
            candidates = [preferred_vendor] + candidates
        else:
            raise ValueError('Unknown vendor {}'.format(preferred_vendor))
    name, path = find_binary_of_command([cls.compiler_dict[x] for x in candidates])
    return (name, path, cls.compiler_name_vendor_mapping[name])