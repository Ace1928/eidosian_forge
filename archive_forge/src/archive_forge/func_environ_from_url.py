import binascii
import io
import os
import re
import sys
import tempfile
import mimetypes
import warnings
from webob.acceptparse import (
from webob.cachecontrol import (
from webob.compat import (
from webob.cookies import RequestCookies
from webob.descriptors import (
from webob.etag import (
from webob.headers import EnvironHeaders
from webob.multidict import (
def environ_from_url(path):
    if SCHEME_RE.search(path):
        scheme, netloc, path, qs, fragment = urlparse.urlsplit(path)
        if fragment:
            raise TypeError('Path cannot contain a fragment (%r)' % fragment)
        if qs:
            path += '?' + qs
        if ':' not in netloc:
            if scheme == 'http':
                netloc += ':80'
            elif scheme == 'https':
                netloc += ':443'
            else:
                raise TypeError('Unknown scheme: %r' % scheme)
    else:
        scheme = 'http'
        netloc = 'localhost:80'
    if path and '?' in path:
        path_info, query_string = path.split('?', 1)
        path_info = url_unquote(path_info)
    else:
        path_info = url_unquote(path)
        query_string = ''
    env = {'REQUEST_METHOD': 'GET', 'SCRIPT_NAME': '', 'PATH_INFO': path_info or '', 'QUERY_STRING': query_string, 'SERVER_NAME': netloc.split(':')[0], 'SERVER_PORT': netloc.split(':')[1], 'HTTP_HOST': netloc, 'SERVER_PROTOCOL': 'HTTP/1.0', 'wsgi.version': (1, 0), 'wsgi.url_scheme': scheme, 'wsgi.input': io.BytesIO(), 'wsgi.errors': sys.stderr, 'wsgi.multithread': False, 'wsgi.multiprocess': False, 'wsgi.run_once': False}
    return env