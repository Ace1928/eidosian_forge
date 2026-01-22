from keystone.common import validation
from keystone.i18n import _
@property
def option_ids(self):
    return set(self._registered_options.keys())