from __future__ import absolute_import, division, print_function
import copy
import datetime
import traceback
import math
import re
from ansible.module_utils.basic import (
from ansible.module_utils.six import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import bigiq_version
from ..module_utils.teem import send_teem
def handle_all_keyword(self):
    if 'all' not in self.want.gather_subset:
        return
    managers = list(self.managers.keys()) + self.want.gather_subset
    managers.remove('all')
    self.want.update({'gather_subset': managers})