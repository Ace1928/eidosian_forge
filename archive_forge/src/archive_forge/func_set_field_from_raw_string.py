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
def set_field_from_raw_string(self, item, raw_string_value, *, preserve_original_field_comment=None, field_comment=None):
    """Sets a field in this paragraph to a given text value

        In many cases, it is better for callers to just use the paragraph as
        if it was a dictionary.  However, this method does enable to you choose
        the field comment (if any) and lets to have a higher degree of control
        over whitespace (on the first line), which can be a reason for using it.

        Example usage:

            >>> example_deb822_paragraph = '''
            ... Package: foo
            ... '''
            >>> dfile = parse_deb822_file(example_deb822_paragraph.splitlines())
            >>> p = next(iter(dfile))
            >>> raw_value = '''
            ... Build-Depends: debhelper-compat (= 12),
            ...                some-other-bd,
            ... # Comment
            ...                another-bd,
            ... '''.lstrip()  # Remove leading newline, but *not* the trailing newline
            >>> fname, new_value = raw_value.split(':', 1)
            >>> p.set_field_from_raw_string(fname, new_value)
            >>> print(p.dump(), end='')
            Package: foo
            Build-Depends: debhelper-compat (= 12),
                           some-other-bd,
            # Comment
                           another-bd,
            >>> # Format preserved

        :param item: Name of the field to set.  If the paragraph already
          contains the field, then it will be replaced.  Otherwise, it is
          added to the end of the paragraph.
          Note this can be a "paragraph key", which enables you to control
          *which* instance of a field is being replaced (in case of duplicate
          fields).
        :param raw_string_value: The text to use as the value.  The text must
          be valid deb822 syntax and is used *exactly* as it is given.
          Accordingly, multi-line values must include mandatory leading space
          on continuation lines, newlines after the value, etc. On the
          flip-side, any optional space or comments will be included.

          Note that the first line will *never* be read as a comment (if the
          first line of the value starts with a "#" then it will result
          in "Field-Name:#..." which is parsed as a value starting with "#"
          rather than a comment).
        :param preserve_original_field_comment: If True, then if there is an
          existing field and that has a comment, then the comment will remain
          after this operation.  This is the default is the `field_comment`
          parameter is omitted.
          Note that if the parameter is True and the item is ambiguous, this
          will raise an AmbiguousDeb822FieldKeyError.  When the parameter is
          omitted, the ambiguity is resolved automatically and if the resolved
          field has a comment then that will be preserved (assuming
          field_comment is None).
        :param field_comment: If not None, add or replace the comment for
          the field.  Each string in the list will become one comment
          line (inserted directly before the field name). Will appear in the
          same order as they do in the list.

          If you want complete control over the formatting of the comments,
          then ensure that each line start with "#" and end with "\\n" before
          the call.  Otherwise, leading/trailing whitespace is normalized
          and the missing "#"/"\\n" character is inserted.
        """
    new_content = []
    if preserve_original_field_comment is not None:
        if field_comment is not None:
            raise ValueError('The "preserve_original_field_comment" conflicts with "field_comment" parameter')
    elif field_comment is not None:
        if not isinstance(field_comment, Deb822CommentElement):
            new_content.extend((_format_comment(x) for x in field_comment))
            field_comment = None
        preserve_original_field_comment = False
    field_name, _, _ = _unpack_key(item)
    cased_field_name = field_name
    try:
        original = self.get_kvpair_element(item, use_get=True)
    except AmbiguousDeb822FieldKeyError:
        if preserve_original_field_comment:
            raise
        original = self.get_kvpair_element((field_name, 0), use_get=True)
    if preserve_original_field_comment is None:
        preserve_original_field_comment = True
    if original:
        cased_field_name = original.field_name
    raw = ':'.join((cased_field_name, raw_string_value))
    raw_lines = raw.splitlines(keepends=True)
    for i, line in enumerate(raw_lines, start=1):
        if not line.endswith('\n'):
            raise ValueError('Line {i} in new value was missing trailing newline'.format(i=i))
        if i != 1 and line[0] not in (' ', '\t', '#'):
            msg = 'Line {i} in new value was invalid.  It must either start with " " space (continuation line) or "#" (comment line). The line started with "{line}"'
            raise ValueError(msg.format(i=i, line=line[0]))
    if len(raw_lines) > 1 and raw_lines[-1].startswith('#'):
        raise ValueError('The last line in a value field cannot be a comment')
    new_content.extend(raw_lines)
    deb822_file = parse_deb822_file(iter(new_content))
    error_token = deb822_file.find_first_error_element()
    if error_token:
        raise ValueError('Syntax error in new field value for ' + field_name)
    paragraph = next(iter(deb822_file))
    assert isinstance(paragraph, Deb822NoDuplicateFieldsParagraphElement)
    value = paragraph.get_kvpair_element(field_name)
    assert value is not None
    if preserve_original_field_comment:
        if original:
            value.comment_element = original.comment_element
            original.comment_element = None
    elif field_comment is not None:
        value.comment_element = field_comment
    self.set_kvpair_element(item, value)