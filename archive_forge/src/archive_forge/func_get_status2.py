from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
def get_status2(self, vm):
    state = vm.info()[0]
    return VIRT_STATE_NAME_MAP.get(state, 'unknown')