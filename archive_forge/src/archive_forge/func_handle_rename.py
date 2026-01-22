from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.infinidat.infinibox.plugins.module_utils.infinibox import (
def handle_rename(module):
    """ Handle rename state """
    switch_name = module.params['switch_name']
    new_switch_name = module.params['new_switch_name']
    switch_result = find_switch_by_name(module)
    switch_id = switch_result['id']
    path = f'fc/switches/{switch_id}'
    data = {'name': new_switch_name}
    try:
        system = get_system(module)
        rename_result = system.api.put(path=path, data=data).get_result()
    except APICommandFailed as err:
        msg = f'Cannot rename fc switch {switch_name}: {err}'
        module.exit_json(msg=msg)
    result = dict(changed=True, msg=f'FC switch renamed from {switch_name} to {new_switch_name}')
    result = merge_two_dicts(result, rename_result)
    module.exit_json(**result)