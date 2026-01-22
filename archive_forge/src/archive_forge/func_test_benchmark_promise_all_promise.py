from pytest import raises
import time
from promise import Promise, promisify, is_thenable
def test_benchmark_promise_all_promise(benchmark):
    values = [Promise.resolve(i) for i in range(100000)]

    def create_promise():
        return Promise.all(values)
    result = benchmark(create_promise)
    assert isinstance(result, Promise)
    assert result.get() == list(range(100000))