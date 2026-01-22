from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def hardware_syn_cookie(self):
    result = flatten_boolean(self._values['hardware_syn_cookie'])
    if result == 'yes':
        return 'enabled'
    if result == 'no':
        return 'disabled'