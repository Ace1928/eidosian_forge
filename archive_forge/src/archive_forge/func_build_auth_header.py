import base64
import cgi
import errno
import http.client
import os
import re
import socket
import ssl
import sys
import time
import urllib
import urllib.request
import weakref
from urllib.parse import urlencode, urljoin, urlparse
from ... import __version__ as breezy_version
from ... import config, debug, errors, osutils, trace, transport, ui, urlutils
from ...bzr.smart import medium
from ...trace import mutter, mutter_callsite
from ...transport import ConnectedTransport, NoSuchFile, UnusableRedirect
from . import default_user_agent, ssl
from .response import handle_response
def build_auth_header(self, auth, request):
    uri = urlparse(request.selector).path
    A1 = ('%s:%s:%s' % (auth['user'], auth['realm'], auth['password'])).encode('utf-8')
    A2 = '{}:{}'.format(request.get_method(), uri).encode('utf-8')
    nonce = auth['nonce']
    qop = auth['qop']
    nonce_count = auth['nonce_count'] + 1
    ncvalue = '%08x' % nonce_count
    cnonce = get_new_cnonce(nonce, nonce_count)
    H, KD = get_digest_algorithm_impls(auth.get('algorithm', 'MD5'))
    nonce_data = '{}:{}:{}:{}:{}'.format(nonce, ncvalue, cnonce, qop, H(A2))
    request_digest = KD(H(A1), nonce_data)
    header = 'Digest '
    header += 'username="{}", realm="{}", nonce="{}"'.format(auth['user'], auth['realm'], nonce)
    header += ', uri="%s"' % uri
    header += ', cnonce="{}", nc={}'.format(cnonce, ncvalue)
    header += ', qop="%s"' % qop
    header += ', response="%s"' % request_digest
    opaque = auth.get('opaque', None)
    if opaque:
        header += ', opaque="%s"' % opaque
    if auth.get('algorithm', None):
        header += ', algorithm="%s"' % auth.get('algorithm')
    auth['nonce_count'] = nonce_count
    return header