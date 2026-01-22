import socket
import sys
import threading
from debugpy.common import log
from debugpy.common.util import hide_thread_from_debugger
Accepts TCP connections on the specified host and port, and invokes the
    provided handler function for every new connection.

    Returns the created server socket.
    