from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
from ansible.module_utils.six.moves.urllib.parse import unquote
def vlan_match(self, pgup, userup, vlanlst):
    res = False
    if pgup and userup:
        return True
    for ln in vlanlst:
        if '-' in ln:
            arr = ln.split('-')
            if int(arr[0]) < self.vlan and self.vlan < int(arr[1]):
                res = True
        elif ln == str(self.vlan):
            res = True
    return res