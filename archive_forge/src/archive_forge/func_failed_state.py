from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def failed_state(sd):
    return sd.status in [sdstate.UNKNOWN, sdstate.INACTIVE]