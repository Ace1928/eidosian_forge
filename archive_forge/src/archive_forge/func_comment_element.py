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
@comment_element.setter
def comment_element(self, value):
    if value is not None:
        if not value[-1].text.endswith('\n'):
            raise ValueError('Field comments must end with a newline')
    if self._comment_element:
        self._comment_element.clear_parent_if_parent(self)
    if value is not None:
        value.parent_element = self
    self._comment_element = value