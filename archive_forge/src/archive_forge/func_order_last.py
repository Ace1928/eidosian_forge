import collections.abc
import contextlib
import sys
import textwrap
import weakref
from abc import ABC
from types import TracebackType
from weakref import ReferenceType
from debian._deb822_repro._util import (combine_into_replacement, BufferingIterator,
from debian._deb822_repro.formatter import (
from debian._deb822_repro.tokens import (
from debian._deb822_repro.types import AmbiguousDeb822FieldKeyError, SyntaxOrParseError
from debian._util import (
def order_last(self, field):
    """Re-order the given field so it is "last" in the paragraph"""
    nodes, nodes_being_relocated = self._nodes_being_relocated(field)
    assert len(nodes_being_relocated) == 1 or len(nodes) == len(nodes_being_relocated)
    kvpair_order = self._kvpair_order
    for node in nodes_being_relocated:
        if kvpair_order.tail_node is node:
            continue
        kvpair_order.remove_node(node)
        assert kvpair_order.tail_node is not None
        kvpair_order.insert_node_after(node, kvpair_order.tail_node)
    if len(nodes_being_relocated) == 1 and nodes_being_relocated[0] is not nodes[-1]:
        single_node = nodes_being_relocated[0]
        nodes.remove(single_node)
        nodes.append(single_node)