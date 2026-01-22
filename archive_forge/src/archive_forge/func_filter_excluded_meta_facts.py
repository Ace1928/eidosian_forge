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
def filter_excluded_meta_facts(self):
    gather_subset = set(self.want.gather_subset)
    gather_subset -= {'!all', '!profiles', '!monitors', '!gtm-pools', '!gtm-wide-ips', '!packages'}
    keys = self.managers.keys()
    if '!all' in self.want.gather_subset:
        gather_subset.clear()
    if '!profiles' in self.want.gather_subset:
        gather_subset -= {x for x in keys if '-profiles' in x}
    if '!monitors' in self.want.gather_subset:
        gather_subset -= {x for x in keys if '-monitors' in x}
    if '!gtm-pools' in self.want.gather_subset:
        gather_subset -= {x for x in keys if x.startswith('gtm-') and x.endswith('-pools')}
    if '!gtm-wide-ips' in self.want.gather_subset:
        gather_subset -= {x for x in keys if x.startswith('gtm-') and x.endswith('-wide-ips')}
    if '!packages' in self.want.gather_subset:
        gather_subset -= {'as3', 'do', 'cfe', 'ts'}
    self.want.update({'gather_subset': list(gather_subset)})