from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
def get_bridge(self, entryid):
    return self.find_entry(entryid).bridgeName()