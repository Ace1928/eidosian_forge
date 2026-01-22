from pytest import raises
import time
from promise import Promise, promisify, is_thenable
def test_benchmark_promise_creation_with_resolve(benchmark):
    do_resolve = lambda resolve, reject: resolve(True)

    def create_promise():
        p = Promise(do_resolve)
        return p
    result = benchmark(create_promise).get()
    assert result == True