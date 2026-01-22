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
def construct_object(self, node, deep=False):
    """deep is True when creating an object/mapping recursively,
        in that case want the underlying elements available during construction
        """
    if node in self.constructed_objects:
        return self.constructed_objects[node]
    if deep:
        old_deep = self.deep_construct
        self.deep_construct = True
    if node in self.recursive_objects:
        return self.recursive_objects[node]
    self.recursive_objects[node] = None
    constructor = None
    tag_suffix = None
    if node.tag in self.yaml_constructors:
        constructor = self.yaml_constructors[node.tag]
    else:
        for tag_prefix in self.yaml_multi_constructors:
            if node.tag.startswith(tag_prefix):
                tag_suffix = node.tag[len(tag_prefix):]
                constructor = self.yaml_multi_constructors[tag_prefix]
                break
        else:
            if None in self.yaml_multi_constructors:
                tag_suffix = node.tag
                constructor = self.yaml_multi_constructors[None]
            elif None in self.yaml_constructors:
                constructor = self.yaml_constructors[None]
            elif isinstance(node, ScalarNode):
                constructor = self.__class__.construct_scalar
            elif isinstance(node, SequenceNode):
                constructor = self.__class__.construct_sequence
            elif isinstance(node, MappingNode):
                constructor = self.__class__.construct_mapping
    if tag_suffix is None:
        data = constructor(self, node)
    else:
        data = constructor(self, tag_suffix, node)
    if isinstance(data, types.GeneratorType):
        generator = data
        data = next(generator)
        if self.deep_construct:
            for _dummy in generator:
                pass
        else:
            self.state_generators.append(generator)
    self.constructed_objects[node] = data
    del self.recursive_objects[node]
    if deep:
        self.deep_construct = old_deep
    return data