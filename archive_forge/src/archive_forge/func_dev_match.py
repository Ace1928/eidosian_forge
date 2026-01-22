from __future__ import (absolute_import, division, print_function)
import uuid
from ansible.module_utils.basic import AnsibleModule, env_fallback
from ansible.module_utils.common.text.converters import to_native
def dev_match(a):
    return dev_id is None or a['device']['id'] == dev_id