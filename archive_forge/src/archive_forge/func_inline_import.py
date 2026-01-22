from __future__ import absolute_import, division, print_function
import os
import time
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def inline_import(self):
    params = self.changes.api_params()
    params['name'] = fq_name(self.want.partition, self.want.name)
    if self.want.source:
        params['filename'] = os.path.split(self.want.source)[1]
    uri = 'https://{0}:{1}/mgmt/tm/asm/tasks/import-policy/'.format(self.client.provider['server'], self.client.provider['server_port'])
    if self.want.force:
        params.update(dict(policyReference={'link': self._get_policy_link()}))
        params.pop('name')
    resp = self.client.api.post(uri, json=params)
    try:
        response = resp.json()
    except ValueError as ex:
        raise F5ModuleError(str(ex))
    if resp.status not in [200, 201] or ('code' in response and response['code'] not in [200, 201]):
        raise F5ModuleError(resp.content)
    return response['id']