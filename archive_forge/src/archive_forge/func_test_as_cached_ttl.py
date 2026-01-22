import time
from concurrent.futures import ThreadPoolExecutor
from panel.io.state import state
def test_as_cached_ttl():

    def test_fn(i=[0]):
        i[0] += 1
        return i[0]
    assert state.as_cached('test', test_fn, ttl=0.1) == 1
    time.sleep(0.11)
    assert state.as_cached('test', test_fn, ttl=0.1) == 2