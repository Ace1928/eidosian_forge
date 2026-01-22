import io
import socket
import ssl
from ..exceptions import ProxySchemeUnsupported
from ..packages import six
def selected_alpn_protocol(self):
    return self.sslobj.selected_alpn_protocol()