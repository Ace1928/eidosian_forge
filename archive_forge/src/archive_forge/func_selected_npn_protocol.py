import io
import socket
import ssl
from ..exceptions import ProxySchemeUnsupported
from ..packages import six
def selected_npn_protocol(self):
    return self.sslobj.selected_npn_protocol()