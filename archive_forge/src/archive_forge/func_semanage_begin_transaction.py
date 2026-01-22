from __future__ import absolute_import, division, print_function
import os
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.six import binary_type
from ansible.module_utils._text import to_bytes, to_text
def semanage_begin_transaction(module, handle):
    if semanage.semanage_begin_transaction(handle) < 0:
        semanage.semanage_handle_destroy(handle)
        module.fail_json(msg='Failed to begin semanage transaction')