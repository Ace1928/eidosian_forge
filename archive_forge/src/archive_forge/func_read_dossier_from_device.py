from __future__ import absolute_import, division, print_function
import re
import time
import xml.etree.ElementTree
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import iControlRestSession
from ..module_utils.teem import send_teem
def read_dossier_from_device(self):
    params = dict(command='run', utilCmdArgs='-b "{0}"'.format(self.want.license_key))
    if self.want.addon_keys:
        addons = self.want.addon_keys
        params['utilCmdArgs'] = '-a {0} '.format(addons) + params['utilCmdArgs']
    if self.want.state == 'revoked':
        params['utilCmdArgs'] = '-r ' + params['utilCmdArgs']
    uri = 'https://{0}:{1}/mgmt/tm/util/get-dossier'.format(self.client.provider['server'], self.client.provider['server_port'])
    resp = self.client.api.post(uri, json=params)
    try:
        response = resp.json()
    except ValueError as ex:
        raise F5ModuleError(str(ex))
    if resp.status not in [200, 201] or ('code' in response and response['code'] not in [200, 201]):
        raise F5ModuleError(resp.content)
    try:
        if self.want.state == 'revoked':
            return response['commandResult'][8:]
        else:
            return response['commandResult']
    except Exception:
        return None