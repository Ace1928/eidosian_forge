from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def check_custom_scheduling_policy_properties():
    if self.param('scheduling_policy_properties'):
        current = []
        if entity.custom_scheduling_policy_properties:
            current = [(sp.name, str(sp.value)) for sp in entity.custom_scheduling_policy_properties]
        passed = [(sp.get('name'), str(sp.get('value'))) for sp in self.param('scheduling_policy_properties') if sp]
        for p in passed:
            if p not in current:
                return False
    return True