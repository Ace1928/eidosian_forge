import functools
import inspect
import reprlib
import socket
import subprocess
import sys
import threading
import traceback
def connect_write_pipe(self, protocol_factory, pipe):
    """Register write pipe in event loop.

        protocol_factory should instantiate object with BaseProtocol interface.
        Pipe is file-like object already switched to nonblocking.
        Return pair (transport, protocol), where transport support
        WriteTransport interface."""
    raise NotImplementedError