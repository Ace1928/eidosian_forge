from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def isZonesetActive(self, zsname):
    if zsname == self.activeZSName:
        return True
    return False