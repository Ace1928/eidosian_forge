from __future__ import absolute_import, division, print_function
import os
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.six import binary_type
from ansible.module_utils._text import to_bytes, to_text
def get_boolean_value(module, name):
    state = 0
    try:
        state = selinux.security_get_boolean_active(name)
    except OSError:
        module.fail_json(msg='Failed to determine current state for boolean %s' % name)
    if state == 1:
        return True
    else:
        return False