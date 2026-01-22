import contextlib
import importlib
import sys
@contextlib.contextmanager
def no_third_party_dir_on_path():
    old_path = sys.path
    try:
        sys.path = sys.path[1:]
        yield
    finally:
        sys.path = old_path