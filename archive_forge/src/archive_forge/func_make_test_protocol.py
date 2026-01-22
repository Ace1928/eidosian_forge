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
def make_test_protocol(base):
    dct = {}
    for name in dir(base):
        if name.startswith('__') and name.endswith('__'):
            continue
        dct[name] = MockCallback(return_value=None)
    return type('TestProtocol', (base,) + base.__bases__, dct)()