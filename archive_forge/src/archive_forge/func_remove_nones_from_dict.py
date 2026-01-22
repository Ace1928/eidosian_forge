from __future__ import (absolute_import, division, print_function)
import os
import json
from ansible.module_utils.basic import AnsibleModule, env_fallback
from ansible.module_utils._text import to_text
def remove_nones_from_dict(obj):
    new_obj = {}
    for key in obj:
        value = obj[key]
        if value is not None and value != {} and (value != []):
            new_obj[key] = value
    if not new_obj:
        return None
    return new_obj