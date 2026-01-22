import functools
from oslotest import base as test_base
from oslo_utils import reflection
def test_inner_callable_function(self):

    def a():

        def b():
            pass
        return b
    name = reflection.get_callable_name(a())
    expected_name = '.'.join((__name__, 'GetCallableNameTestExtended', 'test_inner_callable_function', '<locals>', 'a', '<locals>', 'b'))
    self.assertEqual(expected_name, name)