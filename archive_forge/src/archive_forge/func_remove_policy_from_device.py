from __future__ import absolute_import, division, print_function
import re
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def remove_policy_from_device(self):
    uri = 'https://{0}:{1}/mgmt/tm/ltm/policy/{2}'.format(self.client.provider['server'], self.client.provider['server_port'], transform_name(self.want.partition, self.want.name))
    response = self.client.api.delete(uri)
    if response.status == 200:
        return True
    raise F5ModuleError(response.content)