import functools
from oslotest import base as test_base
from oslo_utils import reflection
def test_no_kwargs(self):
    self.assertEqual(False, reflection.accepts_kwargs(mere_function))