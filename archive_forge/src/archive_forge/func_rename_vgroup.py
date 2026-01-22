from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.common import (
def rename_vgroup(module, array):
    changed = True
    if not rename_exists(module, array):
        try:
            if not module.check_mode:
                array.rename_vgroup(module.params['name'], module.params['rename'])
        except Exception:
            module.fail_json(msg='Rename to {0} failed.'.format(module.params['rename']))
    else:
        module.warn('Rename failed. Volume Group {0} already exists.'.format(module.params['rename']))
    module.exit_json(changed=changed)