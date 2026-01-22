from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import load_config
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_connection, rm_config_prefix
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec
def cli_get_netstream_config(self):
    """ Cli get netstream configuration """
    if self.type == 'ip':
        cmd = 'netstream record %s ip' % self.record_name
    else:
        cmd = 'netstream record %s vxlan inner-ip' % self.record_name
    flags = list()
    regular = '| section include %s' % cmd
    flags.append(regular)
    self.netstream_cfg = get_config(self.module, flags)