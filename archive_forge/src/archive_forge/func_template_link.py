from __future__ import absolute_import, division, print_function
import time
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
@property
def template_link(self):
    if self._values['template_link'] is not None:
        return self._values['template_link']
    result = None
    uri = 'https://{0}:{1}/mgmt/tm/asm/policy-templates/'.format(self.client.provider['server'], self.client.provider['server_port'])
    query = '?$filter=name+eq+{0}'.format(self.template.upper())
    resp = self.client.api.get(uri + query)
    try:
        response = resp.json()
    except ValueError as ex:
        raise F5ModuleError(str(ex))
    if resp.status not in [200, 201] or ('code' in response and response['code'] not in [200, 201]):
        raise F5ModuleError(resp.content)
    if 'items' in response and response['items'] != []:
        result = dict(link=response['items'][0]['selfLink'])
    return result