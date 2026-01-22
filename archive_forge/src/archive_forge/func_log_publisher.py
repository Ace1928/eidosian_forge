from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import cmp_simple_list
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def log_publisher(self):
    if self.want.log_publisher is None:
        return None
    if self.want.log_publisher == '' and self.have.log_publisher in [None, 'none']:
        return None
    if self.want.log_publisher == '':
        if self.want.log_profile is None and self.have.log_profile not in [None, 'none']:
            raise F5ModuleError('The log_publisher cannot be removed if log_profile is defined on device.')
    if self.want.log_publisher != self.have.log_publisher:
        return self.want.log_publisher