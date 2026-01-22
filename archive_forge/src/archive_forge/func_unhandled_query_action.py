from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def unhandled_query_action(self):
    if self._values['unhandled_query_action'] is None:
        return None
    elif self._values['unhandled_query_action'] == 'no-error':
        return 'noerror'
    return self._values['unhandled_query_action']