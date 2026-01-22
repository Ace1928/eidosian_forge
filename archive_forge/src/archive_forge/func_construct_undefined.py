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
def construct_undefined(self, node):
    try:
        if isinstance(node, MappingNode):
            data = CommentedMap()
            data._yaml_set_line_col(node.start_mark.line, node.start_mark.column)
            if node.flow_style is True:
                data.fa.set_flow_style()
            elif node.flow_style is False:
                data.fa.set_block_style()
            data.yaml_set_tag(node.tag)
            yield data
            if node.anchor:
                data.yaml_set_anchor(node.anchor)
            self.construct_mapping(node, data)
            return
        elif isinstance(node, ScalarNode):
            data2 = TaggedScalar()
            data2.value = self.construct_scalar(node)
            data2.style = node.style
            data2.yaml_set_tag(node.tag)
            yield data2
            if node.anchor:
                data2.yaml_set_anchor(node.anchor, always_dump=True)
            return
        elif isinstance(node, SequenceNode):
            data3 = CommentedSeq()
            data3._yaml_set_line_col(node.start_mark.line, node.start_mark.column)
            if node.flow_style is True:
                data3.fa.set_flow_style()
            elif node.flow_style is False:
                data3.fa.set_block_style()
            data3.yaml_set_tag(node.tag)
            yield data3
            if node.anchor:
                data3.yaml_set_anchor(node.anchor)
            data3.extend(self.construct_sequence(node))
            return
    except:
        pass
    raise ConstructorError(None, None, 'could not determine a constructor for the tag %r' % utf8(node.tag), node.start_mark)