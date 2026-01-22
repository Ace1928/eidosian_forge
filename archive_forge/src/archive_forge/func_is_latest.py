from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
def is_latest(module, package):
    rc, out, err = module.run_command(['pkg', 'list', '-u', '--', package])
    return bool(int(rc))