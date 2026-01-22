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
@property
def client_addr(self):
    """
        The effective client IP address as a string.  If the
        ``HTTP_X_FORWARDED_FOR`` header exists in the WSGI environ, this
        attribute returns the client IP address present in that header
        (e.g. if the header value is ``192.168.1.1, 192.168.1.2``, the value
        will be ``192.168.1.1``). If no ``HTTP_X_FORWARDED_FOR`` header is
        present in the environ at all, this attribute will return the value
        of the ``REMOTE_ADDR`` header.  If the ``REMOTE_ADDR`` header is
        unset, this attribute will return the value ``None``.

        .. warning::

           It is possible for user agents to put someone else's IP or just
           any string in ``HTTP_X_FORWARDED_FOR`` as it is a normal HTTP
           header. Forward proxies can also provide incorrect values (private
           IP addresses etc).  You cannot "blindly" trust the result of this
           method to provide you with valid data unless you're certain that
           ``HTTP_X_FORWARDED_FOR`` has the correct values.  The WSGI server
           must be behind a trusted proxy for this to be true.
        """
    e = self.environ
    xff = e.get('HTTP_X_FORWARDED_FOR')
    if xff is not None:
        addr = xff.split(',')[0].strip()
    else:
        addr = e.get('REMOTE_ADDR')
    return addr