from __future__ import absolute_import, division, print_function
import time
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import bigiq_version
from ..module_utils.teem import send_teem
def wait_for_initial_license_activation(self):
    count = 0
    uri = 'https://{0}:{1}/mgmt/cm/device/licensing/pool/initial-activation/{2}'.format(self.client.provider['server'], self.client.provider['server_port'], self.want.license_key)
    while count < 3:
        resp = self.client.api.get(uri)
        try:
            response = resp.json()
        except ValueError as ex:
            raise F5ModuleError(str(ex))
        if 'code' in response and response['code'] == 400:
            if 'message' in response:
                raise F5ModuleError(response['message'])
            else:
                raise F5ModuleError(resp._content)
        if response['status'] == 'READY':
            count += 1
        elif response['status'] == 'ACTIVATING_AUTOMATIC_NEED_EULA_ACCEPT':
            uri = response['selfLink'].replace('https://localhost', 'https://{0}:{1}'.format(self.client.provider['server'], self.client.provider['server_port']))
            self.client.api.patch(uri, json=dict(status='ACTIVATING_AUTOMATIC_EULA_ACCEPTED', eulaText=response['eulaText']))
        elif response['status'] == 'ACTIVATION_FAILED':
            raise F5ModuleError(str(response['message']))
        else:
            count = 0
        time.sleep(1)