from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
import os
import re
def get_repo_states(module):
    rc, out, err = module.run_command([DNF_BIN, 'repolist', '--all', '--verbose'], check_rc=True)
    repos = dict()
    last_repo = ''
    for i, line in enumerate(out.split('\n')):
        m = REPO_ID_RE.match(line)
        if m:
            if len(last_repo) > 0:
                module.fail_json(msg='dnf repolist parse failure: parsed another repo id before next status')
            last_repo = m.group(1)
            continue
        m = REPO_STATUS_RE.match(line)
        if m:
            if len(last_repo) == 0:
                module.fail_json(msg='dnf repolist parse failure: parsed status before repo id')
            repos[last_repo] = m.group(1)
            last_repo = ''
    return repos