from __future__ import absolute_import, division, print_function
import time
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def provision_non_dedicated_on_device(self):
    params = self.want.api_params()
    if self.want.module == 'mgmt':
        uri = 'https://{0}:{1}/mgmt/tm/sys/db/provision.extramb/'.format(self.client.provider['server'], self.client.provider['server_port'])
    else:
        uri = 'https://{0}:{1}/mgmt/tm/sys/provision/{2}'.format(self.client.provider['server'], self.client.provider['server_port'], self.want.module)
    resp = self.client.api.patch(uri, json=params)
    try:
        response = resp.json()
    except ValueError as ex:
        raise F5ModuleError(str(ex))
    if 'code' in response and response['code'] in [400, 404]:
        if 'message' in response:
            raise F5ModuleError(response['message'])
        else:
            raise F5ModuleError(resp.content)