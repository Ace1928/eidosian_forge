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
def wait_for_apply_template_task(self, self_link):
    host = 'https://{0}:{1}'.format(self.client.provider['server'], self.client.provider['server_port'])
    uri = self_link.replace('https://localhost', host)
    while True:
        resp = self.client.api.get(uri)
        try:
            response = resp.json()
        except ValueError as ex:
            raise F5ModuleError(str(ex))
        if response['status'] == 'FINISHED' and response.get('currentStep', None) == 'DONE':
            return True
        elif 'errorMessage' in response:
            raise F5ModuleError(response['errorMessage'])
        time.sleep(5)