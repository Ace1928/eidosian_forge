from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def ip_tos_to_client(self):
    if self.want.ip_tos_to_client is None:
        return None
    if self.want.ip_tos_to_client in ['pass-through', 'mimic']:
        if isinstance(self.have.ip_tos_to_client, int):
            return self.want.ip_tos_to_client
    if self.have.ip_tos_to_client in ['pass-through', 'mimic']:
        if isinstance(self.want.ip_tos_to_client, int):
            return self.want.ip_tos_to_client
    if self.want.ip_tos_to_client != self.have.ip_tos_to_client:
        return self.want.ip_tos_to_client