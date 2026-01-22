from __future__ import absolute_import, division, print_function
import time
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import bigiq_version
from ..module_utils.teem import send_teem
@property
def regkey_pool_uuid(self):
    if self._values['regkey_pool_uuid']:
        return self._values['regkey_pool_uuid']
    collection = self.read_current_from_device()
    resource = next((x for x in collection if x.name == self.regkey_pool), None)
    if resource is None:
        raise F5ModuleError('Could not find the specified regkey pool.')
    self._values['regkey_pool_uuid'] = resource.id
    return resource.id