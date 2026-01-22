from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def create_vector_container_on_device(self):
    params = {'name': self.want.profile}
    uri = 'https://{0}:{1}/mgmt/tm/security/dos/profile/{2}/protocol-sip/'.format(self.client.provider['server'], self.client.provider['server_port'], transform_name(self.want.partition, self.want.profile))
    resp = self.client.api.post(uri, json=params)
    try:
        response = resp.json()
    except ValueError as ex:
        raise F5ModuleError(str(ex))
    if resp.status in [200, 201] or ('code' in response and response['code'] in [200, 201]):
        return True
    raise F5ModuleError(resp.content)