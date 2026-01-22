import functools
from oslotest import base as test_base
from oslo_utils import reflection
def test_inner_callable_class(self):
    obj = self.InnerCallableClass()
    name = reflection.get_callable_name(obj.__call__)
    expected_name = '.'.join((__name__, 'GetCallableNameTestExtended', 'InnerCallableClass', '__call__'))
    self.assertEqual(expected_name, name)