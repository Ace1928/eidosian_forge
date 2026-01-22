from __future__ import absolute_import
import hmac
import os
import sys
import warnings
from binascii import hexlify, unhexlify
from hashlib import md5, sha1, sha256
from ..exceptions import (
from ..packages import six
from .url import BRACELESS_IPV6_ADDRZ_RE, IPV4_RE
def wrap_socket(self, socket, server_hostname=None, server_side=False):
    warnings.warn('A true SSLContext object is not available. This prevents urllib3 from configuring SSL appropriately and may cause certain SSL connections to fail. You can upgrade to a newer version of Python to solve this. For more information, see https://urllib3.readthedocs.io/en/1.26.x/advanced-usage.html#ssl-warnings', InsecurePlatformWarning)
    kwargs = {'keyfile': self.keyfile, 'certfile': self.certfile, 'ca_certs': self.ca_certs, 'cert_reqs': self.verify_mode, 'ssl_version': self.protocol, 'server_side': server_side}
    return wrap_socket(socket, ciphers=self.ciphers, **kwargs)