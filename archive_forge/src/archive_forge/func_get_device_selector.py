from __future__ import (absolute_import, division, print_function)
import uuid
from ansible.module_utils.basic import AnsibleModule, env_fallback
from ansible.module_utils.common.text.converters import to_native
def get_device_selector(spec):
    if is_valid_uuid(spec):
        return lambda v: v['id'] == spec
    else:
        return lambda v: v['hostname'] == spec