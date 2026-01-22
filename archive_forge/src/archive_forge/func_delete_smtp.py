from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
def delete_smtp(module, array):
    """Delete SMTP settings"""
    changed = True
    if not module.check_mode:
        try:
            array.set_smtp(sender_domain='', user_name='', password='', relay_host='')
        except Exception:
            module.fail_json(msg='Delete SMTP settigs failed')
    module.exit_json(changed=changed)