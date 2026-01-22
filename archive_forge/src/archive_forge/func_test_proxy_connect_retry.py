import pytest
from urllib3.exceptions import MaxRetryError, NewConnectionError, ProxyError
from urllib3.poolmanager import ProxyManager
from urllib3.util.retry import Retry
from urllib3.util.url import parse_url
from .port_helpers import find_unused_port
def test_proxy_connect_retry(self):
    retry = Retry(total=None, connect=False)
    port = find_unused_port()
    with ProxyManager('http://localhost:{}'.format(port)) as p:
        with pytest.raises(ProxyError) as ei:
            p.urlopen('HEAD', url='http://localhost/', retries=retry)
        assert isinstance(ei.value.original_error, NewConnectionError)
    retry = Retry(total=None, connect=2)
    with ProxyManager('http://localhost:{}'.format(port)) as p:
        with pytest.raises(MaxRetryError) as ei:
            p.urlopen('HEAD', url='http://localhost/', retries=retry)
        assert isinstance(ei.value.reason.original_error, NewConnectionError)