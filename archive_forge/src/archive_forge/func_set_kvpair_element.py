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
def set_kvpair_element(self, key, value):
    key, index, name_token = _unpack_key(key)
    if name_token:
        if name_token is not value.field_token:
            original_nodes = self._kvpair_elements.get(value.field_name)
            original_node = None
            if original_nodes is not None:
                original_node = self._find_node_via_name_token(name_token, original_nodes)
            if original_node is None:
                raise ValueError('Key is a Deb822FieldNameToken, but not *the* Deb822FieldNameToken for the value nor the Deb822FieldNameToken for an existing field in the paragraph')
            assert original_nodes is not None
            index = original_nodes.index(original_node)
        key = value.field_name
    else:
        if key != value.field_name:
            raise ValueError('Cannot insert value under a different field value than field name from its Deb822FieldNameToken implies')
        key = value.field_name
    original_nodes = self._kvpair_elements.get(key)
    if original_nodes is None or not original_nodes:
        if index is not None and index != 0:
            msg = 'Cannot replace field ({key}, {index}) as the field does not exist in the first place.  Please index-less key or ({key}, 0) if you want to add the field.'
            raise KeyError(msg.format(key=key, index=index))
        node = self._kvpair_order.append(value)
        if key not in self._kvpair_elements:
            self._kvpair_elements[key] = [node]
        else:
            self._kvpair_elements[key].append(node)
        return
    replace_all = False
    if index is None:
        replace_all = True
        node = original_nodes[0]
        if len(original_nodes) != 1:
            self._kvpair_elements[key] = [node]
    else:
        node = original_nodes[index]
    node.value.parent_element = None
    value.parent_element = self
    node.value = value
    if replace_all and len(original_nodes) != 1:
        for n in original_nodes[1:]:
            n.value.parent_element = None
            self._kvpair_order.remove_node(n)