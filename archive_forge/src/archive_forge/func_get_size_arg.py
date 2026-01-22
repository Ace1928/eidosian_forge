import array
import enum
import functools
import logging
import struct
import sys
from typing import List, NamedTuple, Optional, Tuple
import torch
def get_size_arg(self, jitval):
    ctype, value = self.get_constant_value(jitval)
    if ctype.kind() == 'ListType':
        assert ctype.getElementType().kind() == 'IntType'
        return value
    raise Exception(f"Can't handle size arg of type '{ctype!r}' for '{jitval!r}'")