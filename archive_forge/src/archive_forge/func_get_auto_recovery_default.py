from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def get_auto_recovery_default(module):
    auto = False
    data = run_commands(module, ['show inventory | json'])[0]
    pid = data['TABLE_inv']['ROW_inv'][0]['productid']
    if re.search('N7K', pid):
        auto = True
    elif re.search('N9K', pid):
        data = run_commands(module, ['show hardware | json'])[0]
        ver = data['kickstart_ver_str']
        if re.search('7.0\\(3\\)F', ver):
            auto = True
    return auto