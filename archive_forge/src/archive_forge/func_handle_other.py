import logging
import re
import sys
import warnings
from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.exceptions import UnsupportedAlgorithm
from requests.auth import AuthBase
from requests.models import Response
from requests.compat import urlparse, StringIO
from requests.structures import CaseInsensitiveDict
from requests.cookies import cookiejar_from_dict
from requests.packages.urllib3 import HTTPResponse
from .exceptions import MutualAuthenticationError, KerberosExchangeError
def handle_other(self, response):
    """Handles all responses with the exception of 401s.

        This is necessary so that we can authenticate responses if requested"""
    log.debug('handle_other(): Handling: %d' % response.status_code)
    if self.mutual_authentication in (REQUIRED, OPTIONAL) and (not self.auth_done):
        is_http_error = response.status_code >= 400
        if _negotiate_value(response) is not None:
            log.debug('handle_other(): Authenticating the server')
            if not self.authenticate_server(response):
                log.error('handle_other(): Mutual authentication failed')
                raise MutualAuthenticationError('Unable to authenticate {0}'.format(response))
            log.debug('handle_other(): returning {0}'.format(response))
            self.auth_done = True
            return response
        elif is_http_error or self.mutual_authentication == OPTIONAL:
            if not response.ok:
                log.error('handle_other(): Mutual authentication unavailable on {0} response'.format(response.status_code))
            if self.mutual_authentication == REQUIRED and self.sanitize_mutual_error_response:
                return SanitizedResponse(response)
            else:
                return response
        else:
            log.error('handle_other(): Mutual authentication failed')
            raise MutualAuthenticationError('Unable to authenticate {0}'.format(response))
    else:
        log.debug('handle_other(): returning {0}'.format(response))
        return response