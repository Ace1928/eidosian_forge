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
def get_user_password(self, auth):
    """Ask user for a password if none is already available.

        :param auth: authentication info gathered so far (from the initial url
            and then during dialog with the server).
        """
    auth_conf = config.AuthenticationConfig()
    user = auth.get('user', None)
    password = auth.get('password', None)
    realm = auth['realm']
    port = auth.get('port', None)
    if user is None:
        user = auth_conf.get_user(auth['protocol'], auth['host'], port=port, path=auth['path'], realm=realm, ask=True, prompt=self.build_username_prompt(auth))
    if user is not None and password is None:
        password = auth_conf.get_password(auth['protocol'], auth['host'], user, port=port, path=auth['path'], realm=realm, prompt=self.build_password_prompt(auth))
    return (user, password)