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
def sort_fields(self, key=None):
    """Re-order all fields

        :param key: Provide a key function (same semantics as for sorted).   Keep in mind that
          the module preserve the cases for field names - in generally, callers are recommended
          to use "lower()" to normalize the case.
        """
    if key is None:
        key = default_field_sort_key
    key_impl = key

    def _actual_key(kvpair):
        return key_impl(kvpair.field_name)
    for last_kvpair in reversed(self._kvpair_order):
        last_kvpair.value_element.add_final_newline_if_missing()
        break
    sorted_kvpair_list = sorted(self._kvpair_order, key=_actual_key)
    self._kvpair_order = LinkedList()
    self._kvpair_elements = {}
    self._init_kvpair_fields(sorted_kvpair_list)