import operator
from debian._deb822_repro._util import BufferingIterator
from debian._deb822_repro.tokens import Deb822Token
def one_value_per_line_formatter(indentation, trailing_separator=True, immediate_empty_line=False):
    """Provide a simple formatter that can handle indentation and trailing separators

    All formatters returned by this function puts exactly one value per line.  This
    pattern is commonly seen in the "Depends" field and similar fields of
    debian/control files.

    :param indentation: Either the literal string "FIELD_NAME_LENGTH" or a positive
    integer, which determines the indentation for fields.  If it is an integer,
    then a fixed indentation is used (notably the value 1 ensures the shortest
    possible indentation).  Otherwise, if it is "FIELD_NAME_LENGTH", then the
    indentation is set such that it aligns the values based on the field name.
    :param trailing_separator: If True, then the last value will have a trailing
    separator token (e.g., ",") after it.
    :param immediate_empty_line: Whether the value should always start with an
    empty line.  If True, then the result becomes something like "Field:
 value".

    """
    if indentation != 'FIELD_NAME_LENGTH' and indentation < 1:
        raise ValueError('indentation must be at least 1 (or "FIELD_NAME_LENGTH")')

    def _formatter(name, sep_token, formatter_tokens):
        if indentation == 'FIELD_NAME_LENGTH':
            indent_len = len(name) + 2
        else:
            indent_len = indentation
        indent = ' ' * indent_len
        emitted_first_line = False
        tok_iter = BufferingIterator(formatter_tokens)
        is_value = operator.attrgetter('is_value')
        if immediate_empty_line:
            emitted_first_line = True
            yield '\n'
        for t in tok_iter:
            if t.is_comment:
                if not emitted_first_line:
                    yield '\n'
                yield t
            elif t.is_value:
                if not emitted_first_line:
                    yield ' '
                else:
                    yield indent
                yield t
                if not sep_token.is_whitespace and (trailing_separator or tok_iter.peek_find(is_value)):
                    yield sep_token
                yield '\n'
            else:
                continue
            emitted_first_line = True
    return _formatter