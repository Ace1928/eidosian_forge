import contextlib
import io
import socket
from unittest import mock
import netaddr
import netifaces
from oslotest import base as test_base
from oslo_utils import netutils
@contextlib.contextmanager
def mock_file_content(content):
    yield io.StringIO(content)