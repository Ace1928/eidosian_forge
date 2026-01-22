from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def per_source_ip_mitigation_threshold(self):
    if self._values['per_source_ip_mitigation_threshold'] in [None, 'infinite']:
        return self._values['per_source_ip_mitigation_threshold']
    return int(self._values['per_source_ip_mitigation_threshold'])