from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def auto_phone_home(self):
    if self._values['auto_phone_home'] == 'enabled':
        return True
    elif self._values['auto_phone_home'] == 'disabled':
        return False