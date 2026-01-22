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
def construct_rt_sequence(self, node, seqtyp, deep=False):
    if not isinstance(node, SequenceNode):
        raise ConstructorError(None, None, 'expected a sequence node, but found %s' % node.id, node.start_mark)
    ret_val = []
    if node.comment:
        seqtyp._yaml_add_comment(node.comment[:2])
        if len(node.comment) > 2:
            seqtyp.yaml_end_comment_extend(node.comment[2], clear=True)
    if node.anchor:
        from ruamel.yaml.serializer import templated_id
        if not templated_id(node.anchor):
            seqtyp.yaml_set_anchor(node.anchor)
    for idx, child in enumerate(node.value):
        ret_val.append(self.construct_object(child, deep=deep))
        if child.comment:
            seqtyp._yaml_add_comment(child.comment, key=idx)
        seqtyp._yaml_set_idx_line_col(idx, [child.start_mark.line, child.start_mark.column])
    return ret_val