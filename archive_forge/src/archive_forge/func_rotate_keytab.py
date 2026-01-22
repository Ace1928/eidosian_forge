from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def rotate_keytab(module, blade):
    """Rotate keytab"""
    changed = True
    account = Reference(name=list(blade.get_active_directory().items)[0].name, resource_type='active-directory')
    keytab = KeytabPost(source=account)
    if not module.check_mode:
        res = blade.post_keytabs(keytab=keytab, name_prefixes=module.params['prefix'])
        if res.status_code != 200:
            module.fail_json(msg='Failed to rotate AD account keytabs, prefix {0}.'.format(module.params['prefix']))
    module.exit_json(changed=changed)