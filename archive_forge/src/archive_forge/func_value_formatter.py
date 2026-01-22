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
def value_formatter(self, formatter, force_reformat=False):
    """Use a custom formatter when formatting the value

        :param formatter: A formatter (see debian._deb822_repro.formatter.format_field
          for details)
        :param force_reformat: If True, always reformat the field even if there are
          no (other) changes performed.  By default, fields are only reformatted if
          they are changed.
        """
    self._formatter = formatter
    self._format_preserve_original_formatting = False
    if force_reformat:
        self._changed = True