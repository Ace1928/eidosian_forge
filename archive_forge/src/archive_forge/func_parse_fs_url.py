from __future__ import absolute_import, print_function, unicode_literals
import typing
import collections
import re
import six
from six.moves.urllib.parse import parse_qs, unquote
from .errors import ParseError
def parse_fs_url(fs_url):
    """Parse a Filesystem URL and return a `ParseResult`.

    Arguments:
        fs_url (str): A filesystem URL.

    Returns:
        ~fs.opener.parse.ParseResult: a parse result instance.

    Raises:
        ~fs.errors.ParseError: if the FS URL is not valid.

    """
    match = _RE_FS_URL.match(fs_url)
    if match is None:
        raise ParseError('{!r} is not a fs2 url'.format(fs_url))
    fs_name, credentials, url1, url2, path = match.groups()
    if not credentials:
        username = None
        password = None
        url = url2
    else:
        username, _, password = credentials.partition(':')
        username = unquote(username)
        password = unquote(password)
        url = url1
    url, has_qs, qs = url.partition('?')
    resource = unquote(url)
    if has_qs:
        _params = parse_qs(qs, keep_blank_values=True)
        params = {k: unquote(v[0]) for k, v in six.iteritems(_params)}
    else:
        params = {}
    return ParseResult(fs_name, username, password, resource, params, path)