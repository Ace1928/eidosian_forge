from __future__ import absolute_import, division, print_function
import re
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.ipaddress import is_valid_ip
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def provided_username(self):
    if self.want.username:
        return self.username
    if self.want.provider.get('user', None):
        return self.provider.get('user')
    if self.module.params.get('user', None):
        return self.module.params.get('user')