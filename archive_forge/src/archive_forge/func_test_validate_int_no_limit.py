import testtools
from neutronclient.common import exceptions
from neutronclient.common import validators
def test_validate_int_no_limit(self):
    self._test_validate_int(0, min_value=None, max_value=None)
    self._test_validate_int(1, min_value=None, max_value=None)
    self._test_validate_int(10, min_value=None, max_value=None)
    self._test_validate_int(11, min_value=None, max_value=None)
    self._test_validate_int_error('abc', 'attr1 "abc" should be an integer.', min_value=None, max_value=None)