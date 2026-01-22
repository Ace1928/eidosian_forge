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
def do_open(self, http_class, request, first_try=True):
    """See urllib.request.AbstractHTTPHandler.do_open for the general idea.

        The request will be retried once if it fails.
        """
    connection = request.connection
    if connection is None:
        raise AssertionError('Cannot process a request without a connection')
    headers = {}
    headers.update(request.header_items())
    headers.update(request.unredirected_hdrs)
    headers = {name.title(): val for name, val in headers.items()}
    try:
        method = request.get_method()
        url = request.selector
        connection._send_request(method, url, request.data, headers, encode_chunked=headers.get('Transfer-Encoding') == 'chunked')
        if 'http' in debug.debug_flags:
            trace.mutter('> {} {}'.format(method, url))
            hdrs = []
            for k, v in headers.items():
                if k in ('Authorization', 'Proxy-Authorization'):
                    v = '<masked>'
                hdrs.append('{}: {}'.format(k, v))
            trace.mutter('> ' + '\n> '.join(hdrs) + '\n')
        if self._debuglevel >= 1:
            print('Request sent: [%r] from (%s)' % (request, request.connection.sock.getsockname()))
        response = connection.getresponse()
        convert_to_addinfourl = True
    except (ssl.SSLError, ssl.CertificateError):
        raise
    except (socket.gaierror, http.client.BadStatusLine, http.client.UnknownProtocol, OSError, http.client.HTTPException):
        response = self.retry_or_raise(http_class, request, first_try)
        convert_to_addinfourl = False
    response.msg = response.reason
    return response
    if self._debuglevel >= 2:
        print('Receives response: %r' % response)
        print('  For: {!r}({!r})'.format(request.get_method(), request.get_full_url()))
    if convert_to_addinfourl:
        req = request
        r = response
        r.recv = r.read
        fp = socket._fileobject(r, bufsize=65536)
        resp = urllib.request.addinfourl(fp, r.msg, req.get_full_url())
        resp.code = r.status
        resp.msg = r.reason
        resp.version = r.version
        if self._debuglevel >= 2:
            print('Create addinfourl: %r' % resp)
            print('  For: {!r}({!r})'.format(request.get_method(), request.get_full_url()))
        if 'http' in debug.debug_flags:
            version = 'HTTP/%d.%d'
            try:
                version = version % (resp.version / 10, resp.version % 10)
            except:
                version = 'HTTP/%r' % resp.version
            trace.mutter('< {} {} {}'.format(version, resp.code, resp.msg))
            hdrs = [h.rstrip('\r\n') for h in resp.info().headers]
            trace.mutter('< ' + '\n< '.join(hdrs) + '\n')
    else:
        resp = response
    return resp