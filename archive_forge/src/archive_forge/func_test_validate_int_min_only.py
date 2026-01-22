import testtools
from neutronclient.common import exceptions
from neutronclient.common import validators
def test_validate_int_min_only(self):
    self._test_validate_int(1, max_value=None)
    self._test_validate_int(10, max_value=None)
    self._test_validate_int(11, max_value=None)
    self._test_validate_int_error(0, 'attr1 "0" should be an integer greater than or equal to 1.', max_value=None)