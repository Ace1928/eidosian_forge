from time import sleep
from concurrent.futures import ThreadPoolExecutor
from promise import Promise
from operator import mul
def promise_factorial(n):
    if n < 2:
        return 1
    sleep(0.02)
    a = executor.submit(promise_factorial, n - 1)

    def promise_then(r):
        return mul(r, n)
    return Promise.resolve(a).then(promise_then)