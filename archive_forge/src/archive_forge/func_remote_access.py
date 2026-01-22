from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def remote_access(self):
    result = flatten_boolean(self._values['remote_access'])
    if result == 'yes':
        return 'disabled'
    elif result == 'no':
        return 'enabled'