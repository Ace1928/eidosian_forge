import functools
import inspect
import reprlib
import socket
import subprocess
import sys
import threading
import traceback
def create_unix_server(self, protocol_factory, path, *, sock=None, backlog=100, ssl=None):
    """A coroutine which creates a UNIX Domain Socket server.

        The return value is a Server object, which can be used to stop
        the service.

        path is a str, representing a file systsem path to bind the
        server socket to.

        sock can optionally be specified in order to use a preexisting
        socket object.

        backlog is the maximum number of queued connections passed to
        listen() (defaults to 100).

        ssl can be set to an SSLContext to enable SSL over the
        accepted connections.
        """
    raise NotImplementedError