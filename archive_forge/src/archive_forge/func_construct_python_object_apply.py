from __future__ import print_function, absolute_import, division
import datetime
import base64
import binascii
import re
import sys
import types
import warnings
from ruamel.yaml.error import (MarkedYAMLError, MarkedYAMLFutureWarning,
from ruamel.yaml.nodes import *                               # NOQA
from ruamel.yaml.nodes import (SequenceNode, MappingNode, ScalarNode)
from ruamel.yaml.compat import (utf8, builtins_module, to_str, PY2, PY3,  # NOQA
from ruamel.yaml.comments import *                               # NOQA
from ruamel.yaml.comments import (CommentedMap, CommentedOrderedMap, CommentedSet,
from ruamel.yaml.scalarstring import (SingleQuotedScalarString, DoubleQuotedScalarString,
from ruamel.yaml.scalarint import ScalarInt, BinaryInt, OctalInt, HexInt, HexCapsInt
from ruamel.yaml.scalarfloat import ScalarFloat
from ruamel.yaml.scalarbool import ScalarBoolean
from ruamel.yaml.timestamp import TimeStamp
from ruamel.yaml.util import RegExp
def construct_python_object_apply(self, suffix, node, newobj=False):
    if isinstance(node, SequenceNode):
        args = self.construct_sequence(node, deep=True)
        kwds = {}
        state = {}
        listitems = []
        dictitems = {}
    else:
        value = self.construct_mapping(node, deep=True)
        args = value.get('args', [])
        kwds = value.get('kwds', {})
        state = value.get('state', {})
        listitems = value.get('listitems', [])
        dictitems = value.get('dictitems', {})
    instance = self.make_python_instance(suffix, node, args, kwds, newobj)
    if bool(state):
        self.set_python_instance_state(instance, state)
    if bool(listitems):
        instance.extend(listitems)
    if bool(dictitems):
        for key in dictitems:
            instance[key] = dictitems[key]
    return instance