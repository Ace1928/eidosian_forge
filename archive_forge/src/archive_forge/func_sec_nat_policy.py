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
def sec_nat_policy(self):
    if self._values['security_nat_policy'] is None:
        return None
    if 'policy' not in self._values['security_nat_policy']:
        return None
    if self._values['security_nat_policy']['policy'] == '':
        return ''
    return fq_name(self.partition, self._values['security_nat_policy']['policy'])