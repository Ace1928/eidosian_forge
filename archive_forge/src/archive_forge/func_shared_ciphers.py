import io
import socket
import ssl
from ..exceptions import ProxySchemeUnsupported
from ..packages import six
def shared_ciphers(self):
    return self.sslobj.shared_ciphers()