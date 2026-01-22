from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def modify_security_ssh_rest(self, modify):
    """
        Updates the SSH server configuration for the specified SVM.
        """
    if self.parameters.get('vserver'):
        if self.svm_uuid is None:
            self.module.fail_json(msg='Error: no uuid found for the SVM')
        api = 'security/ssh/svms'
    else:
        api = 'security/ssh'
    body = {}
    for option in ('ciphers', 'key_exchange_algorithms', 'mac_algorithms', 'max_authentication_retry_count'):
        if option in modify:
            body[option] = modify[option]
    if body:
        dummy, error = rest_generic.patch_async(self.rest_api, api, self.svm_uuid, body)
        if error:
            self.module.fail_json(msg=error)