from concurrent.futures.thread import ThreadPoolExecutor
from traceback import print_tb
import pytest
import portend
import requests
from requests_toolbelt.sessions import BaseUrlSession as Session
from jaraco.context import ExceptionTrap
from cheroot import wsgi
from cheroot._compat import IS_MACOS, IS_WINDOWS
@pytest.mark.flaky(reruns=3, reruns_delay=2)
def test_connection_keepalive(simple_wsgi_server):
    """Test the connection keepalive works (duh)."""
    session = Session(base_url=simple_wsgi_server['url'])
    pooled = requests.adapters.HTTPAdapter(pool_connections=1, pool_maxsize=1000)
    session.mount('http://', pooled)

    def do_request():
        with ExceptionTrap(requests.exceptions.ConnectionError) as trap:
            resp = session.get('info')
            resp.raise_for_status()
        print_tb(trap.tb)
        return bool(trap)
    with ThreadPoolExecutor(max_workers=10 if IS_SLOW_ENV else 50) as pool:
        tasks = [pool.submit(do_request) for n in range(250 if IS_SLOW_ENV else 1000)]
        failures = sum((task.result() for task in tasks))
    session.close()
    assert not failures