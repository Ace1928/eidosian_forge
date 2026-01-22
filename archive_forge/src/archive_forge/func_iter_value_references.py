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
def iter_value_references(self):
    """Iterate over all values in the list (as ValueReferences)

        This is useful for doing inplace modification of the values or even
        streaming removal of field values.  It is in general also more
        efficient when more than one value is updated or removed.
        """
    yield from (ValueReference(cast('LinkedListNode[VE]', n), self._render, self._value_factory, self._remove_node, self._mark_changed) for n in self._token_list.iter_nodes() if isinstance(n.value, self._vtype))