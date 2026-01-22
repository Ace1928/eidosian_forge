from __future__ import absolute_import, division, print_function
import time
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import bigiq_version
from ..module_utils.ipaddress import is_valid_ip
from ..module_utils.teem import send_teem
@property
def ssg_reference(self):
    filter = "name+eq+'{0}'".format(self.service_environment)
    uri = 'https://{0}:{1}/mgmt/cm/cloud/service-scaling-groups/?$filter={2}&$top=1&$select=selfLink'.format(self.client.provider['server'], self.client.provider['server_port'], filter)
    resp = self.client.api.get(uri)
    try:
        response = resp.json()
    except ValueError as ex:
        raise F5ModuleError(str(ex))
    if resp.status == 200 and response['totalItems'] == 0:
        return None
    elif 'code' in response and response['code'] == 400:
        if 'message' in response:
            raise F5ModuleError(response['message'])
        else:
            raise F5ModuleError(resp._content)
    result = dict(link=response['items'][0]['selfLink'])
    return result