import re
from html.entities import name2codepoint
from typing import Iterable, Match, AnyStr, Optional, Pattern, Tuple, Union
from urllib.parse import urljoin
from w3lib.util import to_unicode
from w3lib.url import safe_url_string
from w3lib._types import StrOrBytes
def replace_entities(text: AnyStr, keep: Iterable[str]=(), remove_illegal: bool=True, encoding: str='utf-8') -> str:
    """Remove entities from the given `text` by converting them to their
    corresponding unicode character.

    `text` can be a unicode string or a byte string encoded in the given
    `encoding` (which defaults to 'utf-8').

    If `keep` is passed (with a list of entity names) those entities will
    be kept (they won't be removed).

    It supports both numeric entities (``&#nnnn;`` and ``&#hhhh;``)
    and named entities (such as ``&nbsp;`` or ``&gt;``).

    If `remove_illegal` is ``True``, entities that can't be converted are removed.
    If `remove_illegal` is ``False``, entities that can't be converted are kept "as
    is". For more information see the tests.

    Always returns a unicode string (with the entities removed).

    >>> import w3lib.html
    >>> w3lib.html.replace_entities(b'Price: &pound;100')
    'Price: \\xa3100'
    >>> print(w3lib.html.replace_entities(b'Price: &pound;100'))
    Price: £100
    >>>

    """

    def convert_entity(m: Match[str]) -> str:
        groups = m.groupdict()
        number = None
        if groups.get('dec'):
            number = int(groups['dec'], 10)
        elif groups.get('hex'):
            number = int(groups['hex'], 16)
        elif groups.get('named'):
            entity_name = groups['named']
            if entity_name.lower() in keep:
                return m.group(0)
            else:
                number = name2codepoint.get(entity_name) or name2codepoint.get(entity_name.lower())
        if number is not None:
            try:
                if 128 <= number <= 159:
                    return bytes((number,)).decode('cp1252')
                else:
                    return chr(number)
            except (ValueError, OverflowError):
                pass
        return '' if remove_illegal and groups.get('semicolon') else m.group(0)
    return _ent_re.sub(convert_entity, to_unicode(text, encoding))