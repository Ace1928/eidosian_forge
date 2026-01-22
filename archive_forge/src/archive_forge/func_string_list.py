from __future__ import absolute_import, division, print_function
import os
import re
from contextlib import contextmanager
from struct import Struct
from ansible.module_utils.six import PY3
def string_list(self, value):
    if not isinstance(value, list):
        raise TypeError('Value must be a list of byte string not %s' % type(value))
    writer = _OpensshWriter()
    for s in value:
        writer.string(s)
    self.string(writer.bytes())
    return self