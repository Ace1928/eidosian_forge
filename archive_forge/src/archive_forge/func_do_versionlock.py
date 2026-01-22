from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
import fnmatch
import os
import re
def do_versionlock(module, command, patterns=None, raw=False):
    patterns = [] if not patterns else patterns
    raw_parameter = ['--raw'] if raw else []
    if patterns:
        outs = []
        for p in patterns:
            rc, out, err = module.run_command([DNF_BIN, '-q', 'versionlock', command] + raw_parameter + [p], check_rc=True)
            outs.append(out)
        out = '\n'.join(outs)
    else:
        rc, out, err = module.run_command([DNF_BIN, '-q', 'versionlock', command], check_rc=True)
    return out