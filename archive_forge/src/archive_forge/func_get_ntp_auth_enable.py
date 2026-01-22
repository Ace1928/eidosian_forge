from __future__ import (absolute_import, division, print_function)
import copy
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec, load_config
from ansible.module_utils.connection import exec_command
def get_ntp_auth_enable(self):
    """Get ntp authentication enable state"""
    flags = list()
    exp = '| exclude undo | include ntp authentication'
    flags.append(exp)
    config = self.get_config(flags)
    auth_en = re.findall('.*ntp\\s*authentication\\s*enable.*', config)
    if auth_en:
        self.ntp_auth_conf['authentication'] = 'enable'
    else:
        self.ntp_auth_conf['authentication'] = 'disable'