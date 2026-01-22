from __future__ import (absolute_import, division, print_function)
import re
import copy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec, load_config
from ansible.module_utils.connection import exec_command
def prase_jumboframe_para(self, config_str):
    """prase_jumboframe_para"""
    interface_cli = 'interface %s' % self.interface.replace(' ', '').lower()
    if config_str.find(interface_cli) == -1:
        self.module.fail_json(msg='Error: Interface does not exist.')
    try:
        npos1 = config_str.index('jumboframe enable')
    except ValueError:
        return [9216, 1518]
    try:
        npos2 = config_str.index('\n', npos1)
        config_str_tmp = config_str[npos1:npos2]
    except ValueError:
        config_str_tmp = config_str[npos1:]
    return re.findall('([0-9]+)', config_str_tmp)