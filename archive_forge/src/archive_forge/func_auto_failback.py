from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def auto_failback(self):
    result = self._values['auto_failback']
    if result == 'true':
        return 'yes'
    if result == 'false':
        return 'no'
    return None