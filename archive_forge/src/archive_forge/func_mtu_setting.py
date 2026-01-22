from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_text
import re
@property
def mtu_setting(self):
    if self.type == 'infiniband':
        return 'infiniband.mtu'
    else:
        return '802-3-ethernet.mtu'