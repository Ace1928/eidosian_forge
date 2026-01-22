from __future__ import (absolute_import, division, print_function)
import uuid
from ansible.module_utils.basic import AnsibleModule, env_fallback
from ansible.module_utils.common.text.converters import to_native
def get_attached_dev_ids(volume_dict):
    if len(volume_dict['attachments']) == 0:
        return []
    else:
        return [a['device']['id'] for a in volume_dict['attachments']]