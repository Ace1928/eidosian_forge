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
def check_mapping_key(self, node, key_node, mapping, key, value):
    """return True if key is unique"""
    if key in mapping:
        if not self.allow_duplicate_keys:
            mk = mapping.get(key)
            if PY2:
                if isinstance(key, unicode):
                    key = key.encode('utf-8')
                if isinstance(value, unicode):
                    value = value.encode('utf-8')
                if isinstance(mk, unicode):
                    mk = mk.encode('utf-8')
            args = ['while constructing a mapping', node.start_mark, 'found duplicate key "{}" with value "{}" (original value: "{}")'.format(key, value, mk), key_node.start_mark, '\n                    To suppress this check see:\n                        http://yaml.readthedocs.io/en/latest/api.html#duplicate-keys\n                    ', '                    Duplicate keys will become an error in future releases, and are errors\n                    by default when using the new API.\n                    ']
            if self.allow_duplicate_keys is None:
                warnings.warn(DuplicateKeyFutureWarning(*args))
            else:
                raise DuplicateKeyError(*args)
        return False
    return True