from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def blacklist_category(self):
    if self._values['blacklist_category'] is None:
        return None
    return fq_name(self.partition, self._values['blacklist_category'])