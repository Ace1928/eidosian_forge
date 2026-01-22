from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def delete_keytab(module, blade):
    """Delete keytab"""
    changed = False
    if blade.get_keytabs(names=[module.params['name']]).status_code == 200:
        changed = True
        if not module.check_mode:
            res = blade.delete_keytabs(names=[module.params['name']])
            if res.status_code != 200:
                module.fail_json(msg='Failed to delete keytab {0}. Error: {1}'.format(module.params['name'], res.errors[0].message))
    module.exit_json(changed=changed)