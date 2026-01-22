from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.urls import build_service_uri
from ..module_utils.teem import send_teem
@property
def device_group(self):
    if self._values['device_group'] not in [None, 'none']:
        result = fq_name(self.partition, self._values['device_group'])
    elif self.param_device_group not in [None, 'none']:
        result = self.param_device_group
    else:
        return None
    if not result.startswith('/Common/'):
        raise F5ModuleError('Device groups can only exist in /Common')
    return result