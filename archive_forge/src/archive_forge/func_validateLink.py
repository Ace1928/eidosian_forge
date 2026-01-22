from __future__ import annotations
from collections.abc import Callable
from contextlib import suppress
import re
from urllib.parse import quote, unquote, urlparse, urlunparse  # noqa: F401
import mdurl
from .. import _punycode
def validateLink(url: str, validator: Callable[[str], bool] | None=None) -> bool:
    """Validate URL link is allowed in output.

    This validator can prohibit more than really needed to prevent XSS.
    It's a tradeoff to keep code simple and to be secure by default.

    Note: url should be normalized at this point, and existing entities decoded.
    """
    if validator is not None:
        return validator(url)
    url = url.strip().lower()
    return bool(GOOD_DATA_RE.search(url)) if BAD_PROTO_RE.search(url) else True