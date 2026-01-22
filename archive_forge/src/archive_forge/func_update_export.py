from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.infinidat.infinibox.plugins.module_utils.infinibox import (
@api_wrapper
def update_export(module, export):
    """ Update existing export """
    if not export:
        raise AssertionError(f'Export {export.get_name()} does not exist and cannot be updated')
    changed = False
    client_list = module.params['client_list']
    if client_list:
        if set(map(transform, unmunchify(export.get_permissions()))) != set(map(transform, client_list)):
            if not module.check_mode:
                export.update_permissions(client_list)
            changed = True
    return changed