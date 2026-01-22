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
def wrap_winrm(self, host, message):
    if not self.winrm_encryption_available:
        raise NotImplementedError('WinRM encryption is not available on the installed version of pykerberos')
    return kerberos.authGSSWinRMEncryptMessage(self.context[host], message)