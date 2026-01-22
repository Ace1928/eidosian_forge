import base64
import codecs
import os
import posixpath
import re
import string
from typing import (
from urllib.parse import (
from urllib.parse import _coerce_args  # type: ignore
from urllib.request import pathname2url, url2pathname
from .util import to_unicode
from ._infra import _ASCII_TAB_OR_NEWLINE, _C0_CONTROL_OR_SPACE
from ._types import AnyUnicodeError, StrOrBytes
from ._url import _SPECIAL_SCHEMES
def safe_url_string(url: StrOrBytes, encoding: str='utf8', path_encoding: str='utf8', quote_path: bool=True) -> str:
    """Return a URL equivalent to *url* that a wide range of web browsers and
    web servers consider valid.

    *url* is parsed according to the rules of the `URL living standard`_,
    and during serialization additional characters are percent-encoded to make
    the URL valid by additional URL standards.

    .. _URL living standard: https://url.spec.whatwg.org/

    The returned URL should be valid by *all* of the following URL standards
    known to be enforced by modern-day web browsers and web servers:

    -   `URL living standard`_

    -   `RFC 3986`_

    -   `RFC 2396`_ and `RFC 2732`_, as interpreted by `Java 8’s java.net.URI
        class`_.

    .. _Java 8’s java.net.URI class: https://docs.oracle.com/javase/8/docs/api/java/net/URI.html
    .. _RFC 2396: https://www.ietf.org/rfc/rfc2396.txt
    .. _RFC 2732: https://www.ietf.org/rfc/rfc2732.txt
    .. _RFC 3986: https://www.ietf.org/rfc/rfc3986.txt

    If a bytes URL is given, it is first converted to `str` using the given
    encoding (which defaults to 'utf-8'). If quote_path is True (default),
    path_encoding ('utf-8' by default) is used to encode URL path component
    which is then quoted. Otherwise, if quote_path is False, path component
    is not encoded or quoted. Given encoding is used for query string
    or form data.

    When passing an encoding, you should use the encoding of the
    original page (the page from which the URL was extracted from).

    Calling this function on an already "safe" URL will return the URL
    unmodified.
    """
    decoded = to_unicode(url, encoding=encoding, errors='percentencode')
    parts = urlsplit(_strip(decoded))
    username, password, hostname, port = (parts.username, parts.password, parts.hostname, parts.port)
    netloc_bytes = b''
    if username is not None or password is not None:
        if username is not None:
            safe_username = quote(unquote(username), _USERINFO_SAFEST_CHARS)
            netloc_bytes += safe_username.encode(encoding)
        if password is not None:
            netloc_bytes += b':'
            safe_password = quote(unquote(password), _USERINFO_SAFEST_CHARS)
            netloc_bytes += safe_password.encode(encoding)
        netloc_bytes += b'@'
    if hostname is not None:
        try:
            netloc_bytes += hostname.encode('idna')
        except UnicodeError:
            netloc_bytes += hostname.encode(encoding)
    if port is not None:
        netloc_bytes += b':'
        netloc_bytes += str(port).encode(encoding)
    netloc = netloc_bytes.decode()
    if quote_path:
        path = quote(parts.path.encode(path_encoding), _PATH_SAFEST_CHARS)
    else:
        path = parts.path
    if parts.scheme in _SPECIAL_SCHEMES:
        query = quote(parts.query.encode(encoding), _SPECIAL_QUERY_SAFEST_CHARS)
    else:
        query = quote(parts.query.encode(encoding), _QUERY_SAFEST_CHARS)
    return urlunsplit((parts.scheme, netloc, path, query, quote(parts.fragment.encode(encoding), _FRAGMENT_SAFEST_CHARS)))