from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import cmp_simple_list
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def frame_distribution_hash(self):
    if self._values['frame_distribution_hash'] is None:
        return None
    elif self._values['frame_distribution_hash'] == 'src-dst-ipport':
        return 'source-destination-ip'
    elif self._values['frame_distribution_hash'] == 'src-dst-mac':
        return 'source-destination-mac'
    elif self._values['frame_distribution_hash'] == 'dst-mac':
        return 'destination-mac'