from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.version import (
from datetime import datetime
import time
def generate_vm_dict(array):
    vm_info = {}
    virt_machines = list(array.get_virtual_machines(vm_type='vvol').items)
    for machine in range(0, len(virt_machines)):
        name = virt_machines[machine].name
        vm_info[name] = {'vm_type': virt_machines[machine].vm_type, 'vm_id': virt_machines[machine].vm_id, 'destroyed': virt_machines[machine].destroyed, 'created': virt_machines[machine].created, 'time_remaining': getattr(virt_machines[machine], 'time_remaining', None), 'latest_snapshot_name': getattr(virt_machines[machine].recover_context, 'name', None), 'latest_snapshot_id': getattr(virt_machines[machine].recover_context, 'id', None)}
    return vm_info