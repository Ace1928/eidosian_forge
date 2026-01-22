from __future__ import absolute_import, division, print_function
import os
import re
import traceback
from collections import namedtuple
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.constants import (
from ..module_utils.compare import cmp_simple_list
from ..module_utils.icontrol import (
from ..module_utils.ipaddress import (
from ..module_utils.teem import send_teem
@property
def security_nat_policy(self):
    result = dict()
    if self.want.sec_nat_use_device_policy is not None:
        if self.want.sec_nat_use_device_policy != self.have.sec_nat_use_device_policy:
            result['use_device_policy'] = self.want.sec_nat_use_device_policy
    if self.want.sec_nat_use_rd_policy is not None:
        if self.want.sec_nat_use_rd_policy != self.have.sec_nat_use_rd_policy:
            result['use_route_domain_policy'] = self.want.sec_nat_use_rd_policy
    if self.want.sec_nat_policy is not None:
        if self.want.sec_nat_policy == '' and self.have.sec_nat_policy is None:
            pass
        elif self.want.sec_nat_policy != self.have.sec_nat_policy:
            result['policy'] = self.want.sec_nat_policy
    if result:
        return dict(security_nat_policy=result)