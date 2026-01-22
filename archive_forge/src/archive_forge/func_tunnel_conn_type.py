from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_text
import re
@property
def tunnel_conn_type(self):
    return self.type in ('gre', 'ipip', 'sit')