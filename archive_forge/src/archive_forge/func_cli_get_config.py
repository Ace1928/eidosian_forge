from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import exec_command, load_config, ce_argument_spec
def cli_get_config(self):
    """ Get configure by cli """
    regular = '| include snmp | include contact'
    flags = list()
    flags.append(regular)
    tmp_cfg = self.get_config(flags)
    return tmp_cfg