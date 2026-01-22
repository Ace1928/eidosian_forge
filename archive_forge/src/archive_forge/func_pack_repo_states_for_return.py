from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
import os
import re
def pack_repo_states_for_return(states):
    enabled = []
    disabled = []
    for repo_id in states:
        if states[repo_id] == 'enabled':
            enabled.append(repo_id)
        else:
            disabled.append(repo_id)
    enabled.sort()
    disabled.sort()
    return {'enabled': enabled, 'disabled': disabled}