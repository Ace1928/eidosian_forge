from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import exec_command, load_config, ce_argument_spec
def cli_get_stp_config(self):
    """ Cli get stp configuration """
    flags = ['| section include #\\s*\\n\\s*stp', '| section exclude #\\s*\\n+\\s*stp process \\d+']
    self.stp_cfg = get_config(self.module, flags)