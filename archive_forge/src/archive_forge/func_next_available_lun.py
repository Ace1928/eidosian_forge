from __future__ import (absolute_import, division, print_function)
from ansible.plugins.lookup import LookupBase
from ansible.errors import AnsibleError
def next_available_lun(self, used_luns):
    """Find next available lun numbers."""
    if self.access_volume_lun is not None:
        used_luns.append(self.access_volume_lun)
    lun = 1
    while lun in used_luns:
        lun += 1
    return lun