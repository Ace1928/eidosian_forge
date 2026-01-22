import binascii
import hashlib
import hmac
import ipaddress
import logging
import urllib.parse as urlparse
import warnings
from oauthlib.common import extract_params, safe_string_equals, urldecode
from . import utils
def base_string_uri(uri: str, host: str=None) -> str:
    """
    Calculates the _base string URI_.

    The *base string URI* is one of the components that make up the
     *signature base string*.

    The ``host`` is optional. If provided, it is used to override any host and
    port values in the ``uri``. The value for ``host`` is usually extracted from
    the "Host" request header from the HTTP request. Its value may be just the
    hostname, or the hostname followed by a colon and a TCP/IP port number
    (hostname:port). If a value for the``host`` is provided but it does not
    contain a port number, the default port number is used (i.e. if the ``uri``
    contained a port number, it will be discarded).

    The rules for calculating the *base string URI* are defined in
    section 3.4.1.2`_ of RFC 5849.

    .. _`section 3.4.1.2`: https://tools.ietf.org/html/rfc5849#section-3.4.1.2

    :param uri: URI
    :param host: hostname with optional port number, separated by a colon
    :return: base string URI
    """
    if not isinstance(uri, str):
        raise ValueError('uri must be a string.')
    output = urlparse.urlparse(uri)
    scheme = output.scheme
    hostname = output.hostname
    port = output.port
    path = output.path
    params = output.params
    if not scheme:
        raise ValueError('missing scheme')
    if not path:
        path = '/'
    scheme = scheme.lower()
    if hostname is not None:
        hostname = hostname.lower()
    if host is not None:
        host = host.lower()
        host = f'{scheme}://{host}'
        output = urlparse.urlparse(host)
        hostname = output.hostname
        port = output.port
    if hostname is None:
        raise ValueError('missing host')
    try:
        hostname = ipaddress.ip_address(hostname)
    except ValueError:
        pass
    if isinstance(hostname, ipaddress.IPv6Address):
        hostname = f'[{hostname}]'
    elif isinstance(hostname, ipaddress.IPv4Address):
        hostname = f'{hostname}'
    if port is not None and (not 0 < port <= 65535):
        raise ValueError('port out of range')
    if (scheme, port) in (('http', 80), ('https', 443)):
        netloc = hostname
    elif port:
        netloc = f'{hostname}:{port}'
    else:
        netloc = hostname
    v = urlparse.urlunparse((scheme, netloc, path, params, '', ''))
    return v.replace(' ', '%20')