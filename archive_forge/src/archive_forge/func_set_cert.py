from __future__ import absolute_import
import datetime
import logging
import os
import re
import socket
import warnings
from socket import error as SocketError
from socket import timeout as SocketTimeout
from .packages import six
from .packages.six.moves.http_client import HTTPConnection as _HTTPConnection
from .packages.six.moves.http_client import HTTPException  # noqa: F401
from .util.proxy import create_proxy_ssl_context
from ._collections import HTTPHeaderDict  # noqa (historical, removed in v2)
from ._version import __version__
from .exceptions import (
from .util import SKIP_HEADER, SKIPPABLE_HEADERS, connection
from .util.ssl_ import (
from .util.ssl_match_hostname import CertificateError, match_hostname
def set_cert(self, key_file=None, cert_file=None, cert_reqs=None, key_password=None, ca_certs=None, assert_hostname=None, assert_fingerprint=None, ca_cert_dir=None, ca_cert_data=None):
    """
        This method should only be called once, before the connection is used.
        """
    if cert_reqs is None:
        if self.ssl_context is not None:
            cert_reqs = self.ssl_context.verify_mode
        else:
            cert_reqs = resolve_cert_reqs(None)
    self.key_file = key_file
    self.cert_file = cert_file
    self.cert_reqs = cert_reqs
    self.key_password = key_password
    self.assert_hostname = assert_hostname
    self.assert_fingerprint = assert_fingerprint
    self.ca_certs = ca_certs and os.path.expanduser(ca_certs)
    self.ca_cert_dir = ca_cert_dir and os.path.expanduser(ca_cert_dir)
    self.ca_cert_data = ca_cert_data