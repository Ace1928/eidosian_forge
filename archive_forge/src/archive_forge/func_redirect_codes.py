from __future__ import absolute_import
import logging
from google.auth import exceptions
from google.auth import transport
import httplib2
from six.moves import http_client
@redirect_codes.setter
def redirect_codes(self, value):
    """Proxy to httplib2.Http.redirect_codes."""
    self.http.redirect_codes = value