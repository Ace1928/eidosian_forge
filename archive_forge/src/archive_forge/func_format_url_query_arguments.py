from __future__ import annotations
import logging  # isort:skip
import re
from typing import Any, Iterable, overload
from urllib.parse import quote_plus
def format_url_query_arguments(url: str, arguments: dict[str, str] | None=None) -> str:
    """ Format a base URL with optional query arguments

    Args:
        url (str) :
            An base URL to append query arguments to
        arguments (dict or None, optional) :
            A mapping of key/value URL query arguments, or None (default: None)

    Returns:
        str

    """
    if arguments is not None:
        items = (f'{quote_plus(key)}={quote_plus(value)}' for key, value in arguments.items())
        url += '?' + '&'.join(items)
    return url