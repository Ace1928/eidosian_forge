from keystone.common import validation
from keystone.i18n import _
def get_option_by_name(self, name):
    for option in self._registered_options.values():
        if name == option.option_name:
            return option
    return None