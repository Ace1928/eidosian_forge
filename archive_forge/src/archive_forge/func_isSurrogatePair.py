from __future__ import absolute_import, division, unicode_literals
from types import ModuleType
from six import text_type, PY3
def isSurrogatePair(data):
    return len(data) == 2 and ord(data[0]) >= 55296 and (ord(data[0]) <= 56319) and (ord(data[1]) >= 56320) and (ord(data[1]) <= 57343)