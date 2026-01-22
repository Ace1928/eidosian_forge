from contextlib import contextmanager
from io import StringIO
import sys
import os
@contextmanager
def osname(name):
    orig = os.name
    os.name = name
    yield
    os.name = orig