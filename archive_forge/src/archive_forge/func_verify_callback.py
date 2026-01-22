import socket
import ssl
import struct
import OpenSSL
from glanceclient import exc
def verify_callback(host=None):
    """Provide wrapper for do_verify_callback.

    We use a partial around the 'real' verify_callback function
    so that we can stash the host value without holding a
    reference on the VerifiedHTTPSConnection.
    """

    def wrapper(connection, x509, errnum, depth, preverify_ok, host=host):
        return do_verify_callback(connection, x509, errnum, depth, preverify_ok, host=host)
    return wrapper