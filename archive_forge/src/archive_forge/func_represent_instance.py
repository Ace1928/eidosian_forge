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
def represent_instance(self, data):
    cls = data.__class__
    class_name = u'%s.%s' % (cls.__module__, cls.__name__)
    args = None
    state = None
    if hasattr(data, '__getinitargs__'):
        args = list(data.__getinitargs__())
    if hasattr(data, '__getstate__'):
        state = data.__getstate__()
    else:
        state = data.__dict__
    if args is None and isinstance(state, dict):
        return self.represent_mapping(u'tag:yaml.org,2002:python/object:' + class_name, state)
    if isinstance(state, dict) and (not state):
        return self.represent_sequence(u'tag:yaml.org,2002:python/object/new:' + class_name, args)
    value = {}
    if bool(args):
        value['args'] = args
    value['state'] = state
    return self.represent_mapping(u'tag:yaml.org,2002:python/object/new:' + class_name, value)