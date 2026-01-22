from __future__ import absolute_import, unicode_literals
import binascii
import hashlib
import hmac
import logging
from oauthlib.common import (extract_params, safe_string_equals, unicode_type,
from . import utils
def collect_parameters(uri_query='', body=[], headers=None, exclude_oauth_signature=True, with_realm=False):
    """**Parameter Sources**

    Parameters starting with `oauth_` will be unescaped.

    Body parameters must be supplied as a dict, a list of 2-tuples, or a
    formencoded query string.

    Headers must be supplied as a dict.

    Per `section 3.4.1.3.1`_ of the spec.

    For example, the HTTP request::

        POST /request?b5=%3D%253D&a3=a&c%40=&a2=r%20b HTTP/1.1
        Host: example.com
        Content-Type: application/x-www-form-urlencoded
        Authorization: OAuth realm="Example",
            oauth_consumer_key="9djdj82h48djs9d2",
            oauth_token="kkk9d7dh3k39sjv7",
            oauth_signature_method="HMAC-SHA1",
            oauth_timestamp="137131201",
            oauth_nonce="7d8f3e4a",
            oauth_signature="djosJKDKJSD8743243%2Fjdk33klY%3D"

        c2&a3=2+q

    contains the following (fully decoded) parameters used in the
    signature base sting::

        +------------------------+------------------+
        |          Name          |       Value      |
        +------------------------+------------------+
        |           b5           |       =%3D       |
        |           a3           |         a        |
        |           c@           |                  |
        |           a2           |        r b       |
        |   oauth_consumer_key   | 9djdj82h48djs9d2 |
        |       oauth_token      | kkk9d7dh3k39sjv7 |
        | oauth_signature_method |     HMAC-SHA1    |
        |     oauth_timestamp    |     137131201    |
        |       oauth_nonce      |     7d8f3e4a     |
        |           c2           |                  |
        |           a3           |        2 q       |
        +------------------------+------------------+

    Note that the value of "b5" is "=%3D" and not "==".  Both "c@" and
    "c2" have empty values.  While the encoding rules specified in this
    specification for the purpose of constructing the signature base
    string exclude the use of a "+" character (ASCII code 43) to
    represent an encoded space character (ASCII code 32), this practice
    is widely used in "application/x-www-form-urlencoded" encoded values,
    and MUST be properly decoded, as demonstrated by one of the "a3"
    parameter instances (the "a3" parameter is used twice in this
    request).

    .. _`section 3.4.1.3.1`:
    https://tools.ietf.org/html/rfc5849#section-3.4.1.3.1
    """
    headers = headers or {}
    params = []
    if uri_query:
        params.extend(urldecode(uri_query))
    if headers:
        headers_lower = dict(((k.lower(), v) for k, v in headers.items()))
        authorization_header = headers_lower.get('authorization')
        if authorization_header is not None:
            params.extend([i for i in utils.parse_authorization_header(authorization_header) if with_realm or i[0] != 'realm'])
    bodyparams = extract_params(body) or []
    params.extend(bodyparams)
    unescaped_params = []
    for k, v in params:
        if k.startswith('oauth_'):
            v = utils.unescape(v)
        unescaped_params.append((k, v))
    if exclude_oauth_signature:
        unescaped_params = list(filter(lambda i: i[0] != 'oauth_signature', unescaped_params))
    return unescaped_params