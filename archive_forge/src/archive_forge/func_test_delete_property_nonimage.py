from unittest import mock
from glance.domain import proxy
import glance.tests.utils as test_utils
def test_delete_property_nonimage(self):
    self.assertRaises(AssertionError, self._test_method, 'delete_property_atomic', None, 'notimage', 'foo', 'bar')