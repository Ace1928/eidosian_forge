from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import compare_complex_list
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def cmp_hash(self):
    if self._values['cmp_hash'] is None:
        return None
    if self._values['cmp_hash'] in ['source-address', 'src', 'src-ip', 'source']:
        return 'src-ip'
    if self._values['cmp_hash'] in ['destination-address', 'dest', 'dst-ip', 'destination', 'dst']:
        return 'dst-ip'
    else:
        return 'default'