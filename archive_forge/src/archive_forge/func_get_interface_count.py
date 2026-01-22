from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
def get_interface_count(_type, source=None):
    key = (_type, source if _type != 'user' else None)
    if key not in similar_interface_counts:
        similar_interface_counts[key] = 1
    else:
        similar_interface_counts[key] += 1
    return similar_interface_counts[key]