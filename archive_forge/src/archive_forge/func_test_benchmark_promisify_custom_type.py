from pytest import raises
import time
from promise import Promise, promisify, is_thenable
def test_benchmark_promisify_custom_type(benchmark):

    class CustomThenable(object):
        pass
    instance = CustomThenable()

    def create_promise():
        return Promise.resolve(instance)
    result = benchmark(create_promise)
    assert isinstance(result, Promise)
    assert result.get() == instance