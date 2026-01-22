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
def flatten_mapping(self, node):
    """
        This implements the merge key feature http://yaml.org/type/merge.html
        by inserting keys from the merge dict/list of dicts if not yet
        available in this node
        """

    def constructed(value_node):
        if value_node in self.constructed_objects:
            value = self.constructed_objects[value_node]
        else:
            value = self.construct_object(value_node, deep=False)
        return value
    merge_map_list = []
    index = 0
    while index < len(node.value):
        key_node, value_node = node.value[index]
        if key_node.tag == u'tag:yaml.org,2002:merge':
            if merge_map_list:
                if self.allow_duplicate_keys:
                    del node.value[index]
                    index += 1
                    continue
                args = ['while constructing a mapping', node.start_mark, 'found duplicate key "{}"'.format(key_node.value), key_node.start_mark, '\n                        To suppress this check see:\n                           http://yaml.readthedocs.io/en/latest/api.html#duplicate-keys\n                        ', '                        Duplicate keys will become an error in future releases, and are errors\n                        by default when using the new API.\n                        ']
                if self.allow_duplicate_keys is None:
                    warnings.warn(DuplicateKeyFutureWarning(*args))
                else:
                    raise DuplicateKeyError(*args)
            del node.value[index]
            if isinstance(value_node, MappingNode):
                merge_map_list.append((index, constructed(value_node)))
            elif isinstance(value_node, SequenceNode):
                for subnode in value_node.value:
                    if not isinstance(subnode, MappingNode):
                        raise ConstructorError('while constructing a mapping', node.start_mark, 'expected a mapping for merging, but found %s' % subnode.id, subnode.start_mark)
                    merge_map_list.append((index, constructed(subnode)))
            else:
                raise ConstructorError('while constructing a mapping', node.start_mark, 'expected a mapping or list of mappings for merging, but found %s' % value_node.id, value_node.start_mark)
        elif key_node.tag == u'tag:yaml.org,2002:value':
            key_node.tag = u'tag:yaml.org,2002:str'
            index += 1
        else:
            index += 1
    return merge_map_list