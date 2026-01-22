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
def set_field_to_simple_value(self, item, simple_value, *, preserve_original_field_comment=None, field_comment=None):
    """Sets a field in this paragraph to a simple "word" or "phrase"

        In many cases, it is better for callers to just use the paragraph as
        if it was a dictionary.  However, this method does enable to you choose
        the field comment (if any), which can be a reason for using it.

        This is suitable for "simple" fields like "Package".  Example:

            >>> example_deb822_paragraph = '''
            ... Package: foo
            ... '''
            >>> dfile = parse_deb822_file(example_deb822_paragraph.splitlines())
            >>> p = next(iter(dfile))
            >>> p.set_field_to_simple_value("Package", "mscgen")
            >>> p.set_field_to_simple_value("Architecture", "linux-any kfreebsd-any",
            ...                             field_comment=['Only ported to linux and kfreebsd'])
            >>> p.set_field_to_simple_value("Priority", "optional")
            >>> print(p.dump(), end='')
            Package: mscgen
            # Only ported to linux and kfreebsd
            Architecture: linux-any kfreebsd-any
            Priority: optional
            >>> # Values are formatted nicely by default, but it does not work with
            >>> # multi-line values
            >>> p.set_field_to_simple_value("Foo", "bar\\nbin\\n")
            Traceback (most recent call last):
                ...
            ValueError: Cannot use set_field_to_simple_value for values with newlines

        :param item: Name of the field to set.  If the paragraph already
          contains the field, then it will be replaced.  If the field exists,
          then it will preserve its order in the paragraph.  Otherwise, it is
          added to the end of the paragraph.
          Note this can be a "paragraph key", which enables you to control
          *which* instance of a field is being replaced (in case of duplicate
          fields).
        :param simple_value: The text to use as the value.  The value must not
          contain newlines.  Leading and trailing will be stripped but space
          within the value is preserved.  The value cannot contain comments
          (i.e. if the "#" token appears in the value, then it is considered
          a value rather than "start of a comment)
        :param preserve_original_field_comment: See the description for the
          parameter with the same name in the set_field_from_raw_string method.
        :param field_comment: See the description for the parameter with the same
          name in the set_field_from_raw_string method.
        """
    if '\n' in simple_value:
        raise ValueError('Cannot use set_field_to_simple_value for values with newlines')
    stripped = simple_value.strip()
    if stripped:
        raw_value = ' ' + stripped + '\n'
    else:
        raw_value = '\n'
    self.set_field_from_raw_string(item, raw_value, preserve_original_field_comment=preserve_original_field_comment, field_comment=field_comment)