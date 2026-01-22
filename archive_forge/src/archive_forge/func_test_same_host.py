from __future__ import absolute_import
import ssl
from socket import error as SocketError
from ssl import SSLError as BaseSSLError
from test import SHORT_TIMEOUT
import pytest
from mock import Mock
from dummyserver.server import DEFAULT_CA
from urllib3._collections import HTTPHeaderDict
from urllib3.connectionpool import (
from urllib3.exceptions import (
from urllib3.packages.six.moves import http_client as httplib
from urllib3.packages.six.moves.http_client import HTTPException
from urllib3.packages.six.moves.queue import Empty
from urllib3.response import HTTPResponse
from urllib3.util.ssl_match_hostname import CertificateError
from urllib3.util.timeout import Timeout
from .test_response import MockChunkedEncodingResponse, MockSock
@pytest.mark.parametrize('a, b', [('http://google.com/', '/'), ('http://google.com/', 'http://google.com/'), ('http://google.com/', 'http://google.com'), ('http://google.com/', 'http://google.com/abra/cadabra'), ('http://google.com:42/', 'http://google.com:42/abracadabra'), ('http://google.com:80/', 'http://google.com/abracadabra'), ('http://google.com/', 'http://google.com:80/abracadabra'), ('https://google.com:443/', 'https://google.com/abracadabra'), ('https://google.com/', 'https://google.com:443/abracadabra'), ('http://[2607:f8b0:4005:805::200e%25eth0]/', 'http://[2607:f8b0:4005:805::200e%eth0]/'), ('https://[2607:f8b0:4005:805::200e%25eth0]:443/', 'https://[2607:f8b0:4005:805::200e%eth0]:443/'), ('http://[::1]/', 'http://[::1]'), ('http://[2001:558:fc00:200:f816:3eff:fef9:b954%lo]/', 'http://[2001:558:fc00:200:f816:3eff:fef9:b954%25lo]')])
def test_same_host(self, a, b):
    with connection_from_url(a) as c:
        assert c.is_same_host(b)