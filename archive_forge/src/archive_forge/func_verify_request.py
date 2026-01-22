import errno
import socket
import socketserver
import sys
import threading
from breezy import cethread, errors, osutils, transport, urlutils
from breezy.bzr.smart import medium, server
from breezy.transport import chroot, pathfilter
def verify_request(self, request, client_address):
    """Verify the request.

        Return True if we should proceed with this request, False if we should
        not even touch a single byte in the socket ! This is useful when we
        stop the server with a dummy last connection.
        """
    return self.serving