from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import string_types
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def self_allow_defaults(self):
    if self.want.self_allow_defaults is None:
        return None
    if self.want.self_allow_defaults == 'none' and self.have.self_allow_defaults is None:
        return None
    if self.want.self_allow_defaults in ['all', 'none']:
        if isinstance(self.have.self_allow_defaults, string_types):
            if self.want.self_allow_defaults != self.have.self_allow_defaults:
                return self.want.self_allow_defaults
            else:
                return None
        if isinstance(self.have.self_allow_defaults, list):
            return self.want.self_allow_defaults
    result = cmp_simple_list(self.want.self_allow_defaults, self.have.self_allow_defaults)
    return result