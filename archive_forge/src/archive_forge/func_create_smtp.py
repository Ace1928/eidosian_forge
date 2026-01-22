from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
def create_smtp(module, array):
    """Set SMTP settings"""
    changed = changed_sender = changed_relay = changed_creds = False
    current_smtp = array.get_smtp()
    if module.params['sender_domain'] and current_smtp['sender_domain'] != module.params['sender_domain']:
        changed_sender = True
        if not module.check_mode:
            try:
                array.set_smtp(sender_domain=module.params['sender_domain'])
            except Exception:
                module.fail_json(msg='Set SMTP sender domain failed.')
    if module.params['relay_host'] and current_smtp['relay_host'] != module.params['relay_host']:
        changed_relay = True
        if not module.check_mode:
            try:
                array.set_smtp(relay_host=module.params['relay_host'])
            except Exception:
                module.fail_json(msg='Set SMTP relay host failed.')
    if module.params['user']:
        changed_creds = True
        if not module.check_mode:
            try:
                array.set_smtp(user_name=module.params['user'], password=module.params['password'])
            except Exception:
                module.fail_json(msg='Set SMTP username/password failed.')
    changed = bool(changed_sender or changed_relay or changed_creds)
    module.exit_json(changed=changed)