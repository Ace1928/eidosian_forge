from __future__ import absolute_import, division, print_function
import re
import time
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def sync_group_to_device(self):
    result = flatten_boolean(self._values['sync_group_to_device'])
    if result == 'yes':
        return True
    if result == 'no':
        return False