import functools
import neutron_lib.callbacks.exceptions as ex
from neutron_lib.tests.unit.exceptions import test_exceptions
def test_inner_exceptions(self):
    key_err = KeyError()
    n_key_err = ex.NotificationError('cb1', key_err)
    err = ex.CallbackFailure([key_err, n_key_err])
    self.assertEqual([key_err, n_key_err.error], err.inner_exceptions)