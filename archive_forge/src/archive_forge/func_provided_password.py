from __future__ import absolute_import, division, print_function
import re
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.ipaddress import is_valid_ip
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def provided_password(self):
    if self.want.password:
        return self.password
    if self.want.provider.get('password', None):
        return self.want.provider.get('password')
    if self.module.params.get('password', None):
        return self.module.params.get('password')