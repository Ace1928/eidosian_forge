from __future__ import annotations
import datetime
from datetime import timedelta as TimeDelta
import binascii
import sys
import types
import warnings
from collections.abc import Hashable, MutableSequence, MutableMapping
from ruamel.yaml.error import (MarkedYAMLError, MarkedYAMLFutureWarning,
from ruamel.yaml.nodes import *                               # NOQA
from ruamel.yaml.nodes import (SequenceNode, MappingNode, ScalarNode)
from ruamel.yaml.compat import (builtins_module, # NOQA
from ruamel.yaml.compat import ordereddict
from ruamel.yaml.tag import Tag
from ruamel.yaml.comments import *                               # NOQA
from ruamel.yaml.comments import (CommentedMap, CommentedOrderedMap, CommentedSet,
from ruamel.yaml.scalarstring import (SingleQuotedScalarString, DoubleQuotedScalarString,
from ruamel.yaml.scalarint import ScalarInt, BinaryInt, OctalInt, HexInt, HexCapsInt
from ruamel.yaml.scalarfloat import ScalarFloat
from ruamel.yaml.scalarbool import ScalarBoolean
from ruamel.yaml.timestamp import TimeStamp
from ruamel.yaml.util import timestamp_regexp, create_timestamp
def construct_unknown(self, node: Any) -> Iterator[Union[CommentedMap, TaggedScalar, CommentedSeq]]:
    try:
        if isinstance(node, MappingNode):
            data = CommentedMap()
            data._yaml_set_line_col(node.start_mark.line, node.start_mark.column)
            if node.flow_style is True:
                data.fa.set_flow_style()
            elif node.flow_style is False:
                data.fa.set_block_style()
            data.yaml_set_ctag(node.ctag)
            yield data
            if node.anchor:
                from ruamel.yaml.serializer import templated_id
                if not templated_id(node.anchor):
                    data.yaml_set_anchor(node.anchor)
            self.construct_mapping(node, data)
            return
        elif isinstance(node, ScalarNode):
            data2 = TaggedScalar()
            data2.value = self.construct_scalar(node)
            data2.style = node.style
            data2.yaml_set_ctag(node.ctag)
            yield data2
            if node.anchor:
                from ruamel.yaml.serializer import templated_id
                if not templated_id(node.anchor):
                    data2.yaml_set_anchor(node.anchor, always_dump=True)
            return
        elif isinstance(node, SequenceNode):
            data3 = CommentedSeq()
            data3._yaml_set_line_col(node.start_mark.line, node.start_mark.column)
            if node.flow_style is True:
                data3.fa.set_flow_style()
            elif node.flow_style is False:
                data3.fa.set_block_style()
            data3.yaml_set_ctag(node.ctag)
            yield data3
            if node.anchor:
                from ruamel.yaml.serializer import templated_id
                if not templated_id(node.anchor):
                    data3.yaml_set_anchor(node.anchor)
            data3.extend(self.construct_sequence(node))
            return
    except:
        pass
    raise ConstructorError(None, None, f'could not determine a constructor for the tag {node.tag!r}', node.start_mark)