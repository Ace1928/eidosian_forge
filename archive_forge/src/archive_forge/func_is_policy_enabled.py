from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
def is_policy_enabled(module, name):
    cmd = '%s list' % AWALL_PATH
    rc, stdout, stderr = module.run_command(cmd)
    if re.search('^%s\\s+enabled' % name, stdout, re.MULTILINE):
        return True
    return False