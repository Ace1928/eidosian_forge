from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import cmp_simple_list
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def log_profile(self):
    if self.want.log_profile is None:
        return None
    if self.want.log_profile == '' and self.have.log_profile in [None, 'none']:
        return None
    if self.want.log_profile == '':
        if self.have.log_publisher not in [None, 'none'] and self.want.log_publisher is None:
            raise F5ModuleError('The log_profile cannot be removed if log_publisher is defined on device.')
    if self.want.log_profile != '':
        if self.want.log_publisher is None and self.have.log_publisher in [None, 'none']:
            raise F5ModuleError('The log_profile cannot be specified without an existing valid log_publisher.')
    if self.want.log_profile != self.have.log_profile:
        return self.want.log_profile