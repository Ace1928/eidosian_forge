from __future__ import absolute_import, division, print_function
import datetime
import math
import re
import time
import traceback
from collections import namedtuple
from ansible.module_utils.basic import (
from ansible.module_utils.parsing.convert_bool import BOOLEANS_TRUE
from ansible.module_utils.six import (
from ansible.module_utils.urls import urlparse
from ipaddress import ip_interface
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.urls import parseStats
from ..module_utils.icontrol import (
from ..module_utils.ipaddress import is_valid_ip
from ..module_utils.teem import send_teem
def handle_gtm_wide_ips_keyword(self):
    if 'gtm-wide-ips' not in self.want.gather_subset:
        return
    keys = self.managers.keys()
    managers = [x for x in keys if x.startswith('gtm-') and x.endswith('-wide-ips')]
    managers += self.want.gather_subset
    managers.remove('gtm-wide-ips')
    self.want.update({'gather_subset': managers})