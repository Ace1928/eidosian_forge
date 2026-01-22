from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def normalize_names_in_device_config(self, name):
    name_map = {'hop-cnt-low': 'hop-cnt-leq-one', 'ip-low-ttl': 'ttl-leq-one'}
    result = name_map.get(name, name)
    return result