from contextlib import contextmanager
from abc import ABC
from abc import abstractmethod
@contextmanager
def current_server(r):
    global _current_server
    remote = _current_server
    _current_server = r
    try:
        yield
    finally:
        _current_server = remote