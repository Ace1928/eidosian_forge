from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def remove_members_in_group_from_device(self):
    uri = 'https://{0}:{1}/mgmt/tm/cm/device-group/{2}/devices/'.format(self.client.provider['server'], self.client.provider['server_port'], self.want.name)
    resp = self.client.api.get(uri)
    try:
        response = resp.json()
    except ValueError as ex:
        raise F5ModuleError(str(ex))
    if resp.status not in [200, 201] or ('code' in response and response['code'] not in [200, 201]):
        raise F5ModuleError(resp.content)
    for item in response['items']:
        new_uri = uri + '{0}'.format(item['name'])
        response = self.client.api.delete(new_uri)
        if response.status == 200:
            return True
        raise F5ModuleError(response.content)