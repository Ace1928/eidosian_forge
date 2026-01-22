import socket
import sys
import threading
from debugpy.common import log
from debugpy.common.util import hide_thread_from_debugger
def shut_down(sock, how=socket.SHUT_RDWR):
    """Shut down the given socket."""
    sock.shutdown(how)