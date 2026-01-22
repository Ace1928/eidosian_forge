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
def order_after(self, field, reference_field):
    """Re-order the given field so appears directly before the reference field in the paragraph

        The reference field must be present.
        """
    nodes, nodes_being_relocated = self._nodes_being_relocated(field)
    assert len(nodes_being_relocated) == 1 or len(nodes) == len(nodes_being_relocated)
    _, reference_nodes = self._nodes_being_relocated(reference_field)
    reference_node = reference_nodes[-1]
    if reference_node in nodes_being_relocated:
        raise ValueError('Cannot re-order a field relative to itself')
    kvpair_order = self._kvpair_order
    for node in reversed(nodes_being_relocated):
        kvpair_order.remove_node(node)
        kvpair_order.insert_node_after(node, reference_node)
    if len(nodes_being_relocated) == 1 and len(nodes) > 1:
        field_name = nodes_being_relocated[0].value.field_name
        self._regenerate_relative_kvapir_order(field_name)