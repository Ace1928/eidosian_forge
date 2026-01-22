from __future__ import absolute_import, division, print_function
import os
import platform
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
def run_acl(module, cmd, check_rc=True):
    try:
        rc, out, err = module.run_command(cmd, check_rc=check_rc)
    except Exception as e:
        module.fail_json(msg=to_native(e))
    lines = []
    for l in out.splitlines():
        if not l.startswith('#'):
            lines.append(l.strip())
    if lines and (not lines[-1].split()):
        return lines[:-1]
    return lines