from keystone.common import validation
from keystone.i18n import _
def get_option_by_id(self, opt_id):
    return self._registered_options.get(opt_id, None)