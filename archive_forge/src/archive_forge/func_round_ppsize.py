from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
def round_ppsize(x, base=16):
    new_size = int(base * round(float(x) / base))
    if new_size < x:
        new_size += base
    return new_size