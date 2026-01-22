from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import remove_default_spec
from ansible_collections.community.network.plugins.module_utils.network.slxos.slxos import get_config, load_config, run_commands
def is_switchport_default(existing):
    """Determines if switchport has a default config based on mode
    Args:
        existing (dict): existing switchport configuration from Ansible mod
    Returns:
        boolean: True if switchport has OOB Layer 2 config, i.e.
           vlan 1 and trunk all and mode is access
    """
    c1 = str(existing['access_vlan']) == '1'
    c2 = str(existing['native_vlan']) == '1'
    c3 = existing['trunk_vlans'] == '1-4094'
    c4 = existing['mode'] == 'access'
    default = c1 and c2 and c3 and c4
    return default