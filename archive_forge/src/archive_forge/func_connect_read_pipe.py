import functools
import inspect
import reprlib
import socket
import subprocess
import sys
import threading
import traceback
def connect_read_pipe(self, protocol_factory, pipe):
    """Register read pipe in event loop. Set the pipe to non-blocking mode.

        protocol_factory should instantiate object with Protocol interface.
        pipe is a file-like object.
        Return pair (transport, protocol), where transport supports the
        ReadTransport interface."""
    raise NotImplementedError