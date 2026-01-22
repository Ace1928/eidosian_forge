from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def partition_access(self):
    if self._values['partition_access'] is None:
        return None
    if self._values['partition_access'] == 'all':
        return 'All'
    return self._values['partition_access']