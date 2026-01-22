from __future__ import absolute_import, division, print_function
import os
import re
import traceback
from collections import namedtuple
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.constants import (
from ..module_utils.compare import cmp_simple_list
from ..module_utils.icontrol import (
from ..module_utils.ipaddress import (
from ..module_utils.teem import send_teem
@property
def fallback_persistence_profile(self):
    if self.want.fallback_persistence_profile is None:
        return None
    if self.want.fallback_persistence_profile == '' and self.have.fallback_persistence_profile is not None:
        return ''
    if self.want.fallback_persistence_profile == '' and self.have.fallback_persistence_profile is None:
        return None
    if self.want.fallback_persistence_profile != self.have.fallback_persistence_profile:
        return self.want.fallback_persistence_profile