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
def generate_request_header(self, response, host, is_preemptive=False):
    """
        Generates the GSSAPI authentication token with kerberos.

        If any GSSAPI step fails, raise KerberosExchangeError
        with failure detail.

        """
    gssflags = kerberos.GSS_C_MUTUAL_FLAG | kerberos.GSS_C_SEQUENCE_FLAG
    if self.delegate:
        gssflags |= kerberos.GSS_C_DELEG_FLAG
    try:
        kerb_stage = 'authGSSClientInit()'
        kerb_host = self.hostname_override if self.hostname_override is not None else host
        kerb_spn = '{0}@{1}'.format(self.service, kerb_host)
        result, self.context[host] = kerberos.authGSSClientInit(kerb_spn, gssflags=gssflags, principal=self.principal)
        if result < 1:
            raise EnvironmentError(result, kerb_stage)
        negotiate_resp_value = '' if is_preemptive else _negotiate_value(response)
        kerb_stage = 'authGSSClientStep()'
        if self.cbt_struct:
            result = kerberos.authGSSClientStep(self.context[host], negotiate_resp_value, channel_bindings=self.cbt_struct)
        else:
            result = kerberos.authGSSClientStep(self.context[host], negotiate_resp_value)
        if result < 0:
            raise EnvironmentError(result, kerb_stage)
        kerb_stage = 'authGSSClientResponse()'
        gss_response = kerberos.authGSSClientResponse(self.context[host])
        return 'Negotiate {0}'.format(gss_response)
    except kerberos.GSSError as error:
        log.exception('generate_request_header(): {0} failed:'.format(kerb_stage))
        log.exception(error)
        raise KerberosExchangeError('%s failed: %s' % (kerb_stage, str(error.args)))
    except EnvironmentError as error:
        if error.errno != result:
            raise
        message = '{0} failed, result: {1}'.format(kerb_stage, result)
        log.error('generate_request_header(): {0}'.format(message))
        raise KerberosExchangeError(message)