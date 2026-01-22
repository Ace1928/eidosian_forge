import re
import codecs
import encodings
from typing import Callable, Match, Optional, Tuple, Union, cast
from w3lib._types import AnyUnicodeError, StrOrBytes
import w3lib.util
def http_content_type_encoding(content_type: Optional[str]) -> Optional[str]:
    """Extract the encoding in the content-type header

    >>> import w3lib.encoding
    >>> w3lib.encoding.http_content_type_encoding("Content-Type: text/html; charset=ISO-8859-4")
    'iso8859-4'

    """
    if content_type:
        match = _HEADER_ENCODING_RE.search(content_type)
        if match:
            return resolve_encoding(match.group(1))
    return None