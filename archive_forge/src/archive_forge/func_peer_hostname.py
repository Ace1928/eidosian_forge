from __future__ import absolute_import, division, print_function
import re
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.ipaddress import is_valid_ip
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def peer_hostname(self):
    if self._values['peer_hostname'] is None:
        return self.peer_server
    regex = re.compile('[^a-zA-Z0-9.\\-_]')
    result = regex.sub('_', self._values['peer_hostname'])
    return result