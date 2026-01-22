import functools
from oslotest import base as test_base
from oslo_utils import reflection
def test_different_simple_callbacks(self):

    def a():
        pass

    def b():
        pass
    self.assertFalse(reflection.is_same_callback(a, b))