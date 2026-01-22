import testtools
from neutronclient.common import exceptions
from neutronclient.common import validators
def test_validate_int_max_only(self):
    self._test_validate_int(0, min_value=None)
    self._test_validate_int(1, min_value=None)
    self._test_validate_int(10, min_value=None)
    self._test_validate_int_error(11, 'attr1 "11" should be an integer smaller than or equal to 10.', min_value=None)