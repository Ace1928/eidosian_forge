import functools
import neutron_lib.callbacks.exceptions as ex
from neutron_lib.tests.unit.exceptions import test_exceptions
def test_callback_failure_with_list(self):
    self._check_exception(ex.CallbackFailure, '1,2,3', errors=[1, 2, 3])