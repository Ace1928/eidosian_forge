import pytest
from functools import wraps, partial
import inspect
import types
@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    if inspect.iscoroutinefunction(pyfuncitem.obj):
        fn = pyfuncitem.obj

        @wraps(fn)
        def wrapper(**kwargs):
            coro = fn(**kwargs)
            try:
                while True:
                    value = coro.send(None)
                    if value != 'mock_sleep':
                        raise RuntimeError('coroutine runner confused: {!r}'.format(value))
            except StopIteration:
                pass
        pyfuncitem.obj = wrapper