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
def represent_folded_scalarstring(self, data):
    tag = None
    style = '>'
    anchor = data.yaml_anchor(any=True)
    for fold_pos in reversed(getattr(data, 'fold_pos', [])):
        if data[fold_pos] == ' ' and (fold_pos > 0 and (not data[fold_pos - 1].isspace())) and (fold_pos < len(data) and (not data[fold_pos + 1].isspace())):
            data = data[:fold_pos] + '\x07' + data[fold_pos:]
    if PY2 and (not isinstance(data, unicode)):
        data = unicode(data, 'ascii')
    tag = u'tag:yaml.org,2002:str'
    return self.represent_scalar(tag, data, style=style, anchor=anchor)