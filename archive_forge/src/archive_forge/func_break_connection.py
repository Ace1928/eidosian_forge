from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, human_to_bytes
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def break_connection(module, blade, target_blade):
    """Break connection between arrays"""
    changed = True
    if not module.check_mode:
        source_blade = blade.arrays.list_arrays().items[0].name
        try:
            if target_blade.management_address is None:
                module.fail_json(msg='Disconnect can only happen from the array that formed the connection')
            blade.array_connections.delete_array_connections(remote_names=[target_blade.remote.name])
        except Exception:
            module.fail_json(msg='Failed to disconnect {0} from {1}.'.format(target_blade.remote.name, source_blade))
    module.exit_json(changed=changed)