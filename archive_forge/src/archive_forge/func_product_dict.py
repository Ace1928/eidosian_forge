import itertools
import types
import unittest
from cupy.testing import _bundle
from cupy.testing import _pytest_impl
def product_dict(*parameters):
    return [{k: v for dic in dicts for k, v in dic.items()} for dicts in itertools.product(*parameters)]