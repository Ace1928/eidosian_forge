from __future__ import absolute_import, division, print_function
from_address:
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.ipaddress import is_valid_ip
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def local_host_name(self):
    if self._values['local_host_name'] is None:
        return None
    if is_valid_ip(self._values['local_host_name']):
        return self._values['local_host_name']
    elif is_valid_hostname(self._values['local_host_name']):
        return str(self._values['local_host_name'])
    raise F5ModuleError("The provided 'local_host_name' value {0} is not a valid IP or hostname".format(str(self._values['local_host_name'])))