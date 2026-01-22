from your HTTP server.
import pprint
import re
import socket
import sys
import time
import traceback
import os
import json
import unittest  # pylint: disable=deprecated-module,preferred-module
import warnings
import functools
import http.client
import urllib.parse
from more_itertools.more import always_iterable
import jaraco.functools
def getPage(self, url, headers=None, method='GET', body=None, protocol=None, raise_subcls=()):
    """Open the url with debugging support.

        Return status, headers, body.

        url should be the identifier passed to the server, typically a
        server-absolute path and query string (sent between method and
        protocol), and should only be an absolute URI if proxy support is
        enabled in the server.

        If the application under test generates absolute URIs, be sure
        to wrap them first with :py:func:`strip_netloc`::

            >>> class MyAppWebCase(WebCase):
            ...     def getPage(url, *args, **kwargs):
            ...         super(MyAppWebCase, self).getPage(
            ...             cheroot.test.webtest.strip_netloc(url),
            ...             *args, **kwargs
            ...         )

        ``raise_subcls`` is passed through to :py:func:`openURL`.
        """
    ServerError.on = False
    if isinstance(url, str):
        url = url.encode('utf-8')
    if isinstance(body, str):
        body = body.encode('utf-8')
    raise_subcls = raise_subcls or ()
    self.url = url
    self.time = None
    start = time.time()
    result = openURL(url, headers, method, body, self.HOST, self.PORT, self.HTTP_CONN, protocol or self.PROTOCOL, raise_subcls=raise_subcls, ssl_context=self.ssl_context)
    self.time = time.time() - start
    self.status, self.headers, self.body = result
    self.cookies = [('Cookie', v) for k, v in self.headers if k.lower() == 'set-cookie']
    if ServerError.on:
        raise ServerError()
    return result