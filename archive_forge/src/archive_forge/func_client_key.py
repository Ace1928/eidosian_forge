from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.ipaddress import is_valid_ip
from ..module_utils.teem import send_teem
@property
def client_key(self):
    if self.have.client_key is None and self.want.client_key == '':
        return None
    if self.have.client_key != self.want.client_key:
        return self.want.client_key