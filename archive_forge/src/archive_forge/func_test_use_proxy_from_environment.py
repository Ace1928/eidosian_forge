import pytest
import threading
import requests
from tests.testserver.server import Server, consume_socket_content
from .utils import override_environ
@pytest.mark.parametrize('var,scheme', _proxy_combos)
def test_use_proxy_from_environment(httpbin, var, scheme):
    url = '{}://httpbin.org'.format(scheme)
    fake_proxy = Server()
    with fake_proxy as (host, port):
        proxy_url = 'socks5://{}:{}'.format(host, port)
        kwargs = {var: proxy_url}
        with override_environ(**kwargs):
            with pytest.raises(requests.exceptions.ConnectionError):
                requests.get(url)
        assert len(fake_proxy.handler_results) == 1
        assert len(fake_proxy.handler_results[0]) > 0