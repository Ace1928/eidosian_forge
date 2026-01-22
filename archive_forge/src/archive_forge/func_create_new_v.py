import itertools
import types
import unittest
from cupy.testing import _bundle
from cupy.testing import _pytest_impl
def create_new_v():
    f = v

    def new_v(self, *args, **kwargs):
        return f(*args, **kwargs)
    return new_v