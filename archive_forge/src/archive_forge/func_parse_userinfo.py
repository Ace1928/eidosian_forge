from __future__ import annotations
import re
import sys
import warnings
from typing import (
from urllib.parse import unquote_plus
from pymongo.client_options import _parse_ssl_options
from pymongo.common import (
from pymongo.errors import ConfigurationError, InvalidURI
from pymongo.srv_resolver import _HAVE_DNSPYTHON, _SrvResolver
from pymongo.typings import _Address
def parse_userinfo(userinfo: str) -> tuple[str, str]:
    """Validates the format of user information in a MongoDB URI.
    Reserved characters that are gen-delimiters (":", "/", "?", "#", "[",
    "]", "@") as per RFC 3986 must be escaped.

    Returns a 2-tuple containing the unescaped username followed
    by the unescaped password.

    :Parameters:
        - `userinfo`: A string of the form <username>:<password>
    """
    if '@' in userinfo or userinfo.count(':') > 1 or _unquoted_percent(userinfo):
        raise InvalidURI('Username and password must be escaped according to RFC 3986, use urllib.parse.quote_plus')
    user, _, passwd = userinfo.partition(':')
    if not user:
        raise InvalidURI('The empty string is not valid username.')
    return (unquote_plus(user), unquote_plus(passwd))