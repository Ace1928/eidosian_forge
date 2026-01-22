from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.config import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def gsa_tcam_check(module):
    """
    global_suppress_arp is an N9k-only command that requires TCAM resources.
    This method checks the current TCAM allocation.
    Note that changing tcam_size requires a switch reboot to take effect.
    """
    cmds = [{'command': 'show hardware access-list tcam region', 'output': 'json'}]
    body = run_commands(module, cmds)
    if body:
        tcam_region = body[0]['TCAM_Region']['TABLE_Sizes']['ROW_Sizes']
        if bool([i for i in tcam_region if i['type'].startswith('Ingress ARP-Ether ACL') and i['tcam_size'] == '0']):
            msg = "'show hardware access-list tcam region' indicates 'ARP-Ether' tcam size is 0 (no allocated resources). " + "'global_suppress_arp' will be rejected by device."
            module.fail_json(msg=msg)