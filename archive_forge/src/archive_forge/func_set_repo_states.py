from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
import os
import re
def set_repo_states(module, repo_ids, state):
    module.run_command([DNF_BIN, 'config-manager', '--set-{0}'.format(state)] + repo_ids, check_rc=True)