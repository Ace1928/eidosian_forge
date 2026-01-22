from __future__ import absolute_import, division, print_function
import re
from time import sleep
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def check_install_in_progress(module, commands, opts):
    for attempt in range(20):
        data = parse_show_install(load_config(module, commands, True, opts))
        if data['install_in_progress']:
            sleep(1)
            continue
        break
    return data