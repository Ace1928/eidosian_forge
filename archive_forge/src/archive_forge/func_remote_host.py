from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.ipaddress import is_valid_ip
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def remote_host(self):
    if is_valid_ip(self._values['remote_host']):
        return self._values['remote_host']
    elif is_valid_hostname(self._values['remote_host']):
        return str(self._values['remote_host'])
    raise F5ModuleError("The provided 'remote_host' is not a valid IP or hostname")