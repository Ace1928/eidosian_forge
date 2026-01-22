import functools
from oslotest import base as test_base
from oslo_utils import reflection
@dummy_decorator
def special_fun(x, y):
    pass