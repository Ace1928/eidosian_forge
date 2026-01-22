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
def evaluate_proxy_bypass(self, host, no_proxy):
    """Check the host against a comma-separated no_proxy list as a string.

        :param host: ``host:port`` being requested

        :param no_proxy: comma-separated list of hosts to access directly.

        :returns: True to skip the proxy, False not to, or None to
            leave it to urllib.
        """
    if no_proxy is None:
        return False
    hhost, hport = splitport(host)
    for domain in no_proxy.split(','):
        domain = domain.strip()
        if domain == '':
            continue
        dhost, dport = splitport(domain)
        if hport == dport or dport is None:
            dhost = dhost.replace('.', '\\.')
            dhost = dhost.replace('*', '.*')
            dhost = dhost.replace('?', '.')
            if re.match(dhost, hhost, re.IGNORECASE):
                return True
    return None