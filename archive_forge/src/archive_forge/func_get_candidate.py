from __future__ import absolute_import, division, print_function
import os
import tempfile
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.config import dumps
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def get_candidate(self):
    candidate = ''
    if self.want.src:
        candidate = self.want.src
    elif self.want.lines:
        candidate_obj = ImishConfig(indent=1)
        parents = self.want.parents or list()
        if self.want.allow_duplicates:
            candidate_obj.add(self.want.lines, parents=parents, duplicates=True)
        else:
            candidate_obj.add(self.want.lines, parents=parents)
        candidate = dumps(candidate_obj, 'raw')
    return candidate