import contextlib
import threading
import time
from taskflow import exceptions as excp
from taskflow.persistence.backends import impl_dir
from taskflow import states
from taskflow.tests import utils as test_utils
from taskflow.utils import persistence_utils as p_utils
from taskflow.utils import threading_utils
@contextlib.contextmanager
def connect_close(*args):
    try:
        for a in args:
            a.connect()
        yield
    finally:
        for a in args:
            a.close()