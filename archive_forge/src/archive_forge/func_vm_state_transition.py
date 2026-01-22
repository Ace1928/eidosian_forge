from __future__ import absolute_import, division, print_function
import json
import os
import re
import tempfile
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
def vm_state_transition(module, uuid, vm_state):
    ret = set_vm_state(module, uuid, vm_state)
    if ret is None:
        return False
    elif ret:
        return True
    else:
        module.fail_json(msg='Failed to set VM {0} to state {1}'.format(uuid, vm_state))