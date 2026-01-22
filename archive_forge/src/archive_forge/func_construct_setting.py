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
def construct_setting(self, node, typ, deep=False):
    if not isinstance(node, MappingNode):
        raise ConstructorError(None, None, 'expected a mapping node, but found %s' % node.id, node.start_mark)
    if node.comment:
        typ._yaml_add_comment(node.comment[:2])
        if len(node.comment) > 2:
            typ.yaml_end_comment_extend(node.comment[2], clear=True)
    if node.anchor:
        from ruamel.yaml.serializer import templated_id
        if not templated_id(node.anchor):
            typ.yaml_set_anchor(node.anchor)
    for key_node, value_node in node.value:
        key = self.construct_object(key_node, deep=True)
        if not isinstance(key, Hashable):
            if isinstance(key, list):
                key = tuple(key)
        if PY2:
            try:
                hash(key)
            except TypeError as exc:
                raise ConstructorError('while constructing a mapping', node.start_mark, 'found unacceptable key (%s)' % exc, key_node.start_mark)
        elif not isinstance(key, Hashable):
            raise ConstructorError('while constructing a mapping', node.start_mark, 'found unhashable key', key_node.start_mark)
        value = self.construct_object(value_node, deep=deep)
        self.check_set_key(node, key_node, typ, key)
        if key_node.comment:
            typ._yaml_add_comment(key_node.comment, key=key)
        if value_node.comment:
            typ._yaml_add_comment(value_node.comment, value=key)
        typ.add(key)