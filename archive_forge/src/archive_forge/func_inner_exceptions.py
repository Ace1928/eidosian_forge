from neutron_lib._i18n import _
from neutron_lib import exceptions
@property
def inner_exceptions(self):
    """The list of unpacked errors for this exception.

        :return: A list of unpacked errors for this exception. An unpacked
            error is the Exception's 'error' attribute if it inherits from
            NotificationError, otherwise it's the exception itself.
        """
    if isinstance(self.errors, list):
        return [self._unpack_if_notification_error(e) for e in self.errors]
    return [self._unpack_if_notification_error(self.errors)]