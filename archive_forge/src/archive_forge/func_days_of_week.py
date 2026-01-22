from __future__ import absolute_import, division, print_function
import re
import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def days_of_week(self):
    return cmp_simple_list(self.want.days_of_week, self.have.days_of_week)