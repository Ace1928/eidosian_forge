from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def time_wait_recycle(self):
    if self._values['time_wait_recycle'] is None:
        return None
    elif self._values['time_wait_recycle'] == 'enabled':
        return 'yes'
    return 'no'