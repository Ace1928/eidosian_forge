import sys
import warnings
from eventlet import greenpool
from eventlet import greenthread
from eventlet import support
from eventlet.green import socket
from eventlet.support import greenlets as greenlet
def wrap_ssl(sock, *a, **kw):
    """Convenience function for converting a regular socket into an
    SSL socket.  Has the same interface as :func:`ssl.wrap_socket`,
    but can also use PyOpenSSL. Though, note that it ignores the
    `cert_reqs`, `ssl_version`, `ca_certs`, `do_handshake_on_connect`,
    and `suppress_ragged_eofs` arguments when using PyOpenSSL.

    The preferred idiom is to call wrap_ssl directly on the creation
    method, e.g., ``wrap_ssl(connect(addr))`` or
    ``wrap_ssl(listen(addr), server_side=True)``. This way there is
    no "naked" socket sitting around to accidentally corrupt the SSL
    session.

    :return Green SSL object.
    """
    return wrap_ssl_impl(sock, *a, **kw)