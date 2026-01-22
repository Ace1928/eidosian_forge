import threading
from contextlib import contextmanager
import pytest
from tornado import ioloop, web
from dummyserver.handlers import TestingApp
from dummyserver.proxy import ProxyHandler
from dummyserver.server import (
from urllib3.connection import HTTPConnection
def assert_header_received(self, received_headers, header_name, expected_value=None):
    header_name = header_name.encode('ascii')
    if expected_value is not None:
        expected_value = expected_value.encode('ascii')
    header_titles = []
    for header in received_headers:
        key, value = header.split(b': ')
        header_titles.append(key)
        if key == header_name and expected_value is not None:
            assert value == expected_value
    assert header_name in header_titles