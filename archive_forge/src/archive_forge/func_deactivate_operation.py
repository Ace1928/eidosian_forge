from __future__ import absolute_import, division, print_function
import time
from copy import deepcopy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def deactivate_operation(module, show_cmd, pkg, flag):
    cmd = 'install deactivate {0} forced'.format(pkg)
    config_cmd_operation(module, cmd)
    validate_operation(module, show_cmd, cmd, pkg, flag)
    return cmd