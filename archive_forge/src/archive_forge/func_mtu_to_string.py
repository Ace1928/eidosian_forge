from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_text
import re
@staticmethod
def mtu_to_string(mtu):
    if not mtu:
        return 'auto'
    else:
        return to_text(mtu)