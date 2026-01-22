from __future__ import absolute_import, division, print_function
import re
from datetime import datetime
from ansible.module_utils.basic import (
from ipaddress import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.ipaddress import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def geo_locations(self):
    if self._values['geo_locations'] is None:
        return None
    result = []
    for x in self._values['geo_locations']:
        if x['region'] is not None and x['region'].strip() != '':
            tmp = '{0}:{1}'.format(x['country'], x['region'])
        else:
            tmp = x['country']
        result.append(tmp)
    result = sorted(result)
    return result