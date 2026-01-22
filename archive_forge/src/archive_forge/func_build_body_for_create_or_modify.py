from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
from ansible_collections.netapp.ontap.plugins.module_utils import rest_vserver
def build_body_for_create_or_modify(self, policy_type, body=None):
    if body is None:
        body = {}
    if 'comment' in self.parameters.keys():
        body['comment'] = self.parameters['comment']
    if 'is_network_compression_enabled' in self.parameters:
        if policy_type == 'sync':
            self.module.fail_json(msg='Error: input parameter network_compression_enabled is not valid for SnapMirror policy type sync')
        body['network_compression_enabled'] = self.parameters['is_network_compression_enabled']
    for option in ('identity_preservation', 'transfer_schedule'):
        if option in self.parameters:
            if policy_type == 'sync':
                self.module.fail_json(msg='Error: %s is only supported with async (async) policy_type, got: %s' % (option, self.parameters['policy_type']))
            body[option] = self.parameters[option]
    return body