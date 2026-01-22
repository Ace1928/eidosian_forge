import pytest
from urllib3.exceptions import MaxRetryError, NewConnectionError, ProxyError
from urllib3.poolmanager import ProxyManager
from urllib3.util.retry import Retry
from urllib3.util.url import parse_url
from .port_helpers import find_unused_port
def test_default_port(self):
    with ProxyManager('http://something') as p:
        assert p.proxy.port == 80
    with ProxyManager('https://something') as p:
        assert p.proxy.port == 443