from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def delete_cert(module, blade):
    """Delete certificate"""
    changed = True
    if not module.check_mode:
        try:
            blade.certificates.delete_certificates(names=[module.params['name']])
        except Exception:
            module.fail_json(msg='Failed to delete certificate {0}.'.format(module.params['name']))
    module.exit_json(changed=changed)