from neutron_lib._i18n import _
from neutron_lib import exceptions
@property
def is_cancellable(self):
    return self._cancellable