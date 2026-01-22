from __future__ import absolute_import, division, print_function
import os
import time
from datetime import datetime
from ansible.module_utils.urls import urlparse
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def remove_iso_from_device(self, type):
    uri = 'https://{0}:{1}/mgmt/tm/sys/software/{2}/{3}'.format(self.client.provider['server'], self.client.provider['server_port'], type, self.want.filename)
    response = self.client.api.delete(uri)
    if response.status == 200:
        return True
    if 'code' in response and response['code'] in [400, 404]:
        if 'message' in response:
            raise F5ModuleError(response['message'])
        else:
            raise F5ModuleError(response.content)