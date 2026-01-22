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
def merge_comments(self, node, comments):
    if comments is None:
        assert hasattr(node, 'comment')
        return node
    if getattr(node, 'comment', None) is not None:
        for idx, val in enumerate(comments):
            if idx >= len(node.comment):
                continue
            nc = node.comment[idx]
            if nc is not None:
                assert val is None or val == nc
                comments[idx] = nc
    node.comment = comments
    return node