from concurrent.futures import ThreadPoolExecutor
from promise import Promise
import time
import weakref
import gc
def function_with_local_type():

    class A:
        pass
    a = A()
    assert a == Promise.resolve(a).get()
    return weakref.ref(A)