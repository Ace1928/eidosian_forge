from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def force_onboard_actions(self):
    """ synchronize and passphrase are not returned in GET so we need to be creative """
    if 'onboard' not in self.parameters:
        return (None, None)
    passphrase = self.na_helper.safe_get(self.parameters, ['onboard', 'passphrase'])
    modify_sync = None
    if self.na_helper.safe_get(self.parameters, ['onboard', 'synchronize']):
        if passphrase is None:
            self.module.fail_json(msg='Error: passphrase is required for synchronize.')
        modify_sync = {'onboard': {'synchronize': True, 'existing_passphrase': passphrase}}
    modify_passphrase = None
    from_passphrase = self.na_helper.safe_get(self.parameters, ['onboard', 'from_passphrase'])
    if passphrase and (not from_passphrase):
        self.module.warn('passphrase is ignored')
    if not passphrase and from_passphrase and (not modify_sync):
        self.module.warn('from_passphrase is ignored')
    if passphrase and from_passphrase and self.is_passphrase_update_required(passphrase, from_passphrase):
        modify_passphrase = {'onboard': {'passphrase': passphrase, 'existing_passphrase': from_passphrase}}
    if modify_passphrase or modify_sync:
        self.na_helper.changed = True
    return (modify_passphrase, modify_sync)