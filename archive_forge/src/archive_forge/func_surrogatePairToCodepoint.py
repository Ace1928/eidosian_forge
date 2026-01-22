from __future__ import absolute_import, division, unicode_literals
from types import ModuleType
from six import text_type, PY3
def surrogatePairToCodepoint(data):
    char_val = 65536 + (ord(data[0]) - 55296) * 1024 + (ord(data[1]) - 56320)
    return char_val