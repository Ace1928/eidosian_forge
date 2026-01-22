from time import sleep
from concurrent.futures import ThreadPoolExecutor
from promise import Promise
from operator import mul
def test_factorial():
    p = promise_factorial(10)
    assert p.get() == 3628800