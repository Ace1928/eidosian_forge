from __future__ import absolute_import, division, print_function
import os
import re
from contextlib import contextmanager
from struct import Struct
from ansible.module_utils.six import PY3
def uint32(self, value):
    if not isinstance(value, int):
        raise TypeError('Value must be of type int not %s' % type(value))
    if value < 0 or value > _UINT32_MAX:
        raise ValueError('Value must be a positive integer less than %s' % _UINT32_MAX)
    self._buff.extend(_UINT32.pack(value))
    return self