from __future__ import absolute_import
import OpenSSL.SSL
from cryptography import x509
from cryptography.hazmat.backends.openssl import backend as openssl_backend
from cryptography.hazmat.backends.openssl.x509 import _Certificate
from io import BytesIO
from socket import error as SocketError
from socket import timeout
import logging
import ssl
import sys
from .. import util
from ..packages import six
from ..util.ssl_ import PROTOCOL_TLS_CLIENT
def recv_into(self, *args, **kwargs):
    try:
        return self.connection.recv_into(*args, **kwargs)
    except OpenSSL.SSL.SysCallError as e:
        if self.suppress_ragged_eofs and e.args == (-1, 'Unexpected EOF'):
            return 0
        else:
            raise SocketError(str(e))
    except OpenSSL.SSL.ZeroReturnError:
        if self.connection.get_shutdown() == OpenSSL.SSL.RECEIVED_SHUTDOWN:
            return 0
        else:
            raise
    except OpenSSL.SSL.WantReadError:
        if not util.wait_for_read(self.socket, self.socket.gettimeout()):
            raise timeout('The read operation timed out')
        else:
            return self.recv_into(*args, **kwargs)
    except OpenSSL.SSL.Error as e:
        raise ssl.SSLError('read error: %r' % e)