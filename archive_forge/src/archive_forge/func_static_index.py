import functools
import unittest
from pecan import expose
from pecan import util
from pecan.compat import getargspec
@staticmethod
@expose()
def static_index(a, b, c=1, *args, **kwargs):
    return 'Hello, World!'