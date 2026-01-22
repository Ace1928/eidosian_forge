from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.config import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (

    global_suppress_arp is an N9k-only command that requires TCAM resources.
    This method checks the current TCAM allocation.
    Note that changing tcam_size requires a switch reboot to take effect.
    