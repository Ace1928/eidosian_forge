from __future__ import (absolute_import, division, print_function)
import copy
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec, load_config
from ansible.module_utils.connection import exec_command
def get_ntp_auth_exist_config(self):
    """Get ntp authentication existed configure"""
    self.get_ntp_auth_enable()
    self.get_ntp_all_auth_keyid()