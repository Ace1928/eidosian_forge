from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
def parse_out(string):
    return re.sub('\\s+', ' ', string).strip()