from sys import maxunicode
from typing import cast, Iterable, Iterator, List, MutableSet, Union, Optional
from .unicode_categories import RAW_UNICODE_CATEGORIES
from .codepoints import CodePoint, code_point_order, code_point_repr, \
def iterparse_character_subset(s: str, expand_ranges: bool=False) -> Iterator[CodePoint]:
    """
    Parses a regex character subset, generating a sequence of code points
    and code points ranges. An unescaped hyphen (-) that is not at the
    start or at the and is interpreted as range specifier.

    :param s: a string representing the character subset.
    :param expand_ranges: if set to `True` then expands character ranges.
    :return: yields integers or couples of integers.
    """
    escaped = False
    on_range = False
    char = ''
    length = len(s)
    subset_index_iterator = iter(range(len(s)))
    for k in subset_index_iterator:
        if k == 0:
            char = s[0]
            if char == '\\':
                escaped = True
            elif char in '[]' and length > 1:
                raise RegexError('bad character %r at position 0' % char)
            elif expand_ranges:
                yield ord(char)
            elif length <= 2 or s[1] != '-':
                yield ord(char)
        elif s[k] == '-':
            if escaped or k == length - 1:
                char = s[k]
                yield ord(char)
                escaped = False
            elif on_range:
                char = s[k]
                yield ord(char)
                on_range = False
            else:
                on_range = True
                k = next(subset_index_iterator)
                end_char = s[k]
                if end_char == '\\' and k < length - 1:
                    if s[k + 1] in '-|.^?*+{}()[]':
                        k = next(subset_index_iterator)
                        end_char = s[k]
                    elif s[k + 1] in 'sSdDiIcCwWpP':
                        msg = "bad character range '%s-\\%s' at position %d: %r"
                        raise RegexError(msg % (char, s[k + 1], k - 2, s))
                if ord(char) > ord(end_char):
                    msg = "bad character range '%s-%s' at position %d: %r"
                    raise RegexError(msg % (char, end_char, k - 2, s))
                elif expand_ranges:
                    yield from range(ord(char) + 1, ord(end_char) + 1)
                else:
                    yield (ord(char), ord(end_char) + 1)
        elif s[k] in '|.^?*+{}()':
            if escaped:
                escaped = False
            on_range = False
            char = s[k]
            yield ord(char)
        elif s[k] in '[]':
            if not escaped and length > 1:
                raise RegexError('bad character %r at position %d' % (s[k], k))
            escaped = on_range = False
            char = s[k]
            if k >= length - 2 or s[k + 1] != '-':
                yield ord(char)
        elif s[k] == '\\':
            if escaped:
                escaped = on_range = False
                char = '\\'
                yield ord(char)
            else:
                escaped = True
        else:
            if escaped:
                escaped = False
                yield ord('\\')
            on_range = False
            char = s[k]
            if k >= length - 2 or s[k + 1] != '-':
                yield ord(char)
    if escaped:
        yield ord('\\')