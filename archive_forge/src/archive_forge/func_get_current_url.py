from __future__ import annotations
import typing as t
from urllib.parse import quote
from .._internal import _plain_int
from ..exceptions import SecurityError
from ..urls import uri_to_iri
def get_current_url(scheme: str, host: str, root_path: str | None=None, path: str | None=None, query_string: bytes | None=None) -> str:
    """Recreate the URL for a request. If an optional part isn't
    provided, it and subsequent parts are not included in the URL.

    The URL is an IRI, not a URI, so it may contain Unicode characters.
    Use :func:`~werkzeug.urls.iri_to_uri` to convert it to ASCII.

    :param scheme: The protocol the request used, like ``"https"``.
    :param host: The host the request was made to. See :func:`get_host`.
    :param root_path: Prefix that the application is mounted under. This
        is prepended to ``path``.
    :param path: The path part of the URL after ``root_path``.
    :param query_string: The portion of the URL after the "?".
    """
    url = [scheme, '://', host]
    if root_path is None:
        url.append('/')
        return uri_to_iri(''.join(url))
    url.append(quote(root_path.rstrip('/'), safe="!$&'()*+,/:;=@%"))
    url.append('/')
    if path is None:
        return uri_to_iri(''.join(url))
    url.append(quote(path.lstrip('/'), safe="!$&'()*+,/:;=@%"))
    if query_string:
        url.append('?')
        url.append(quote(query_string, safe="!$&'()*+,/:;=?@%"))
    return uri_to_iri(''.join(url))