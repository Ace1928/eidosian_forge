import re
from html.entities import name2codepoint
from typing import Iterable, Match, AnyStr, Optional, Pattern, Tuple, Union
from urllib.parse import urljoin
from w3lib.util import to_unicode
from w3lib.url import safe_url_string
from w3lib._types import StrOrBytes
def replace_escape_chars(text: AnyStr, which_ones: Iterable[str]=('\n', '\t', '\r'), replace_by: StrOrBytes='', encoding: Optional[str]=None) -> str:
    """Remove escape characters.

    `which_ones` is a tuple of which escape characters we want to remove.
    By default removes ``\\n``, ``\\t``, ``\\r``.

    `replace_by` is the string to replace the escape characters by.
    It defaults to ``''``, meaning the escape characters are removed.

    """
    utext = to_unicode(text, encoding)
    for ec in which_ones:
        utext = utext.replace(ec, to_unicode(replace_by, encoding))
    return utext