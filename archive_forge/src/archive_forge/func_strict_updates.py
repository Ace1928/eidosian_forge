from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.urls import build_service_uri
from ..module_utils.teem import send_teem
@property
def strict_updates(self):
    if self._values['strict_updates'] is not None:
        result = flatten_boolean(self._values['strict_updates'])
    elif self.param_strict_updates is not None:
        result = flatten_boolean(self.param_strict_updates)
    else:
        return None
    if result == 'yes':
        return 'enabled'
    return 'disabled'