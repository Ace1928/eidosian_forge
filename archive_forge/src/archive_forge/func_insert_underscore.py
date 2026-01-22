from __future__ import print_function, absolute_import, division
from ruamel.yaml.error import *  # NOQA
from ruamel.yaml.nodes import *  # NOQA
from ruamel.yaml.compat import text_type, binary_type, to_unicode, PY2, PY3, ordereddict
from ruamel.yaml.compat import nprint, nprintf  # NOQA
from ruamel.yaml.scalarstring import (
from ruamel.yaml.scalarint import ScalarInt, BinaryInt, OctalInt, HexInt, HexCapsInt
from ruamel.yaml.scalarfloat import ScalarFloat
from ruamel.yaml.scalarbool import ScalarBoolean
from ruamel.yaml.timestamp import TimeStamp
import datetime
import sys
import types
from ruamel.yaml.comments import (
def insert_underscore(self, prefix, s, underscore, anchor=None):
    if underscore is None:
        return self.represent_scalar(u'tag:yaml.org,2002:int', prefix + s, anchor=anchor)
    if underscore[0]:
        sl = list(s)
        pos = len(s) - underscore[0]
        while pos > 0:
            sl.insert(pos, '_')
            pos -= underscore[0]
        s = ''.join(sl)
    if underscore[1]:
        s = '_' + s
    if underscore[2]:
        s += '_'
    return self.represent_scalar(u'tag:yaml.org,2002:int', prefix + s, anchor=anchor)