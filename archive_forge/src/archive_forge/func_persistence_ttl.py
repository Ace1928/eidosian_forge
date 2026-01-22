from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
@property
def persistence_ttl(self):
    if self._values['persistence_ttl'] is None:
        return None
    if 0 <= self._values['persistence_ttl'] <= 4294967295:
        return self._values['persistence_ttl']
    raise F5ModuleError("Valid 'persistence_ttl' must be in range 0 - 4294967295.")