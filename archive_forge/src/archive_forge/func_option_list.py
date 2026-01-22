from __future__ import absolute_import, division, print_function
import os
import re
from contextlib import contextmanager
from struct import Struct
from ansible.module_utils.six import PY3
def option_list(self, value):
    if not isinstance(value, list) or (value and (not isinstance(value[0], tuple))):
        raise TypeError('Value must be a list of tuples')
    writer = _OpensshWriter()
    for name, data in value:
        writer.string(name)
        writer.string(_OpensshWriter().string(data).bytes() if data else bytes())
    self.string(writer.bytes())
    return self