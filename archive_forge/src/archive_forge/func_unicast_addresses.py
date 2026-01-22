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
@property
def unicast_addresses(self):
    if self._values['unicast_addresses'] is None:
        return None
    result = []
    for addr in self._values['unicast_addresses']:
        tmp = {}
        for key in ['effectiveIp', 'effectivePort', 'ip', 'port']:
            if key in addr:
                renamed_key = self.convert(key)
                tmp[renamed_key] = addr.get(key, None)
        if tmp:
            result.append(tmp)
    if result:
        return result