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
def get_running_config(self, config=None):
    contents = self.want.running_config
    if not contents:
        if config:
            contents = config
        else:
            contents = self.read_current_from_device()
    return contents