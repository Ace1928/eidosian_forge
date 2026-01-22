from __future__ import absolute_import, division, print_function
import traceback
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def modify_iscsi_service_state_and_target(self, modify):
    body = {}
    api = 'protocols/san/iscsi/services'
    if 'target_alias' in modify:
        body['target.alias'] = self.parameters['target_alias']
        dummy, error = rest_generic.patch_async(self.rest_api, api, self.uuid, body)
        if error:
            self.module.fail_json(msg='Error modifying iscsi service target alias on vserver %s: %s' % (self.parameters['vserver'], error))