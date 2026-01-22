import re
import sys
from weakref import ReferenceType
import weakref
from debian._util import resolve_ref, _strI
from debian._deb822_repro._util import BufferingIterator
def tokenize_deb822_file(sequence, encoding='utf-8'):
    """Tokenize a deb822 file

    :param sequence: An iterable of lines (a file open for reading will do)
    :param encoding: The encoding to use (this is here to support Deb822-like
       APIs, new code should not use this parameter).
    """
    current_field_name = None
    field_name_cache = {}

    def _normalize_input(s):
        for x in s:
            if isinstance(x, bytes):
                x = x.decode(encoding)
            if not x.endswith('\n'):
                x += '\n'
            yield x
    text_stream = BufferingIterator(_normalize_input(sequence))
    for line in text_stream:
        if line.isspace():
            if current_field_name:
                current_field_name = None
            r = list(text_stream.takewhile(str.isspace))
            if r:
                line += ''.join(r)
            yield Deb822WhitespaceToken(sys.intern(line))
            continue
        if line[0] == '#':
            yield Deb822CommentToken(line)
            continue
        if line[0] in (' ', '\t'):
            if current_field_name is not None:
                leading = sys.intern(line[0])
                line = line[1:-1]
                yield Deb822ValueContinuationToken(leading)
                yield Deb822ValueToken(line)
                yield Deb822NewlineAfterValueToken()
            else:
                yield Deb822ErrorToken(line)
            continue
        field_line_match = _RE_FIELD_LINE.match(line)
        if field_line_match:
            field_name, _, space_before, value, space_after = field_line_match.groups()
            current_field_name = field_name_cache.get(field_name)
            if value is None or value == '':
                space_after = space_before + space_after if space_after else space_before
                space_before = ''
            if space_after:
                if space_after.endswith('\n'):
                    space_after = space_after[:-1]
            if current_field_name is None:
                field_name = sys.intern(field_name)
                current_field_name = _strI(field_name)
                field_name_cache[field_name] = current_field_name
            del field_name
            yield Deb822FieldNameToken(current_field_name)
            yield Deb822FieldSeparatorToken()
            if space_before:
                yield Deb822WhitespaceToken(sys.intern(space_before))
            if value:
                yield Deb822ValueToken(value)
            if space_after:
                yield Deb822WhitespaceToken(sys.intern(space_after))
            yield Deb822NewlineAfterValueToken()
        else:
            yield Deb822ErrorToken(line)