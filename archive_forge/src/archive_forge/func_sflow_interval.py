from __future__ import absolute_import, division, print_function
import copy
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import cmp_str_with_none
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def sflow_interval(self):
    if self._values['sflow'] is None:
        return None
    if self._values['sflow']['poll_interval'] is None:
        return None
    if 0 <= self._values['sflow']['poll_interval'] <= 4294967295:
        return self._values['sflow']['poll_interval']
    raise F5ModuleError("Valid 'poll_interval' must be in range 0 - 4294967295.")