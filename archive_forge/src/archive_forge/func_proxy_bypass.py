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
def proxy_bypass(self, host):
    """Check if host should be proxied or not.

        :returns: True to skip the proxy, False otherwise.
        """
    no_proxy = self.get_proxy_env_var('no', default_to=None)
    bypass = self.evaluate_proxy_bypass(host, no_proxy)
    if bypass is None:
        return urllib.request.proxy_bypass(host)
    else:
        return bypass