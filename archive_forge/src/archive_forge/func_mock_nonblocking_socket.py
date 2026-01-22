import collections
import contextlib
import io
import logging
import os
import re
import socket
import socketserver
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock
from http.server import HTTPServer
from wsgiref.simple_server import WSGIRequestHandler, WSGIServer
from . import base_events
from . import events
from . import futures
from . import selectors
from . import tasks
from .coroutines import coroutine
from .log import logger
def mock_nonblocking_socket():
    """Create a mock of a non-blocking socket."""
    sock = mock.Mock(socket.socket)
    sock.gettimeout.return_value = 0.0
    return sock