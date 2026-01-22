import re
from html.entities import name2codepoint
from typing import Iterable, Match, AnyStr, Optional, Pattern, Tuple, Union
from urllib.parse import urljoin
from w3lib.util import to_unicode
from w3lib.url import safe_url_string
from w3lib._types import StrOrBytes
def unquote_markup(text: AnyStr, keep: Iterable[str]=(), remove_illegal: bool=True, encoding: Optional[str]=None) -> str:
    """
    This function receives markup as a text (always a unicode string or
    a UTF-8 encoded string) and does the following:

    1. removes entities (except the ones in `keep`) from any part of it
        that is not inside a CDATA
    2. searches for CDATAs and extracts their text (if any) without modifying it.
    3. removes the found CDATAs

    """

    def _get_fragments(txt: str, pattern: Pattern[str]) -> Iterable[Union[str, Match[str]]]:
        offset = 0
        for match in pattern.finditer(txt):
            match_s, match_e = match.span(1)
            yield txt[offset:match_s]
            yield match
            offset = match_e
        yield txt[offset:]
    utext = to_unicode(text, encoding)
    ret_text = ''
    for fragment in _get_fragments(utext, _cdata_re):
        if isinstance(fragment, str):
            ret_text += replace_entities(fragment, keep=keep, remove_illegal=remove_illegal)
        else:
            ret_text += fragment.group('cdata_d')
    return ret_text