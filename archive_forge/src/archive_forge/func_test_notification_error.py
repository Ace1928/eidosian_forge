import functools
import neutron_lib.callbacks.exceptions as ex
from neutron_lib.tests.unit.exceptions import test_exceptions
def test_notification_error(self):
    """Test that correct message is created for this error class."""
    error = ex.NotificationError('abc', 'boom')
    self.assertEqual('Callback abc failed with "boom"', str(error))