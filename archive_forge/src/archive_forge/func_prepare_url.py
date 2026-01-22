import datetime
import sys
import encodings.idna
from urllib3.fields import RequestField
from urllib3.filepost import encode_multipart_formdata
from urllib3.util import parse_url
from urllib3.exceptions import (
from io import UnsupportedOperation
from .hooks import default_hooks
from .structures import CaseInsensitiveDict
from .auth import HTTPBasicAuth
from .cookies import cookiejar_from_dict, get_cookie_header, _copy_cookie_jar
from .exceptions import (
from .exceptions import JSONDecodeError as RequestsJSONDecodeError
from .exceptions import SSLError as RequestsSSLError
from ._internal_utils import to_native_string, unicode_is_ascii
from .utils import (
from .compat import (
from .compat import json as complexjson
from .status_codes import codes
def prepare_url(self, url, params):
    """Prepares the given HTTP URL."""
    if isinstance(url, bytes):
        url = url.decode('utf8')
    else:
        url = unicode(url) if is_py2 else str(url)
    url = url.lstrip()
    if ':' in url and (not url.lower().startswith('http')):
        self.url = url
        return
    try:
        scheme, auth, host, port, path, query, fragment = parse_url(url)
    except LocationParseError as e:
        raise InvalidURL(*e.args)
    if not scheme:
        error = 'Invalid URL {0!r}: No scheme supplied. Perhaps you meant http://{0}?'
        error = error.format(to_native_string(url, 'utf8'))
        raise MissingSchema(error)
    if not host:
        raise InvalidURL('Invalid URL %r: No host supplied' % url)
    if not unicode_is_ascii(host):
        try:
            host = self._get_idna_encoded_host(host)
        except UnicodeError:
            raise InvalidURL('URL has an invalid label.')
    elif host.startswith((u'*', u'.')):
        raise InvalidURL('URL has an invalid label.')
    netloc = auth or ''
    if netloc:
        netloc += '@'
    netloc += host
    if port:
        netloc += ':' + str(port)
    if not path:
        path = '/'
    if is_py2:
        if isinstance(scheme, str):
            scheme = scheme.encode('utf-8')
        if isinstance(netloc, str):
            netloc = netloc.encode('utf-8')
        if isinstance(path, str):
            path = path.encode('utf-8')
        if isinstance(query, str):
            query = query.encode('utf-8')
        if isinstance(fragment, str):
            fragment = fragment.encode('utf-8')
    if isinstance(params, (str, bytes)):
        params = to_native_string(params)
    enc_params = self._encode_params(params)
    if enc_params:
        if query:
            query = '%s&%s' % (query, enc_params)
        else:
            query = enc_params
    url = requote_uri(urlunparse([scheme, netloc, path, None, query, fragment]))
    self.url = url