import os
import ssl
import sys
from ... import config
from ... import version_string as breezy_version
def default_ca_certs():
    if sys.platform == 'win32':
        return os.path.join(os.path.dirname(sys.executable), 'cacert.pem')
    elif sys.platform == 'darwin':
        pass
    else:
        for path in _ssl_ca_certs_known_locations:
            if os.path.exists(path):
                return path
    return _ssl_ca_certs_known_locations[0]