import os
import platform
import socket
import ssl
import typing
import _ssl  # type: ignore[import]
from ._ssl_constants import (
def inject_into_ssl() -> None:
    """Injects the :class:`truststore.SSLContext` into the ``ssl``
    module by replacing :class:`ssl.SSLContext`.
    """
    setattr(ssl, 'SSLContext', SSLContext)
    try:
        import pip._vendor.urllib3.util.ssl_ as urllib3_ssl
        setattr(urllib3_ssl, 'SSLContext', SSLContext)
    except ImportError:
        pass