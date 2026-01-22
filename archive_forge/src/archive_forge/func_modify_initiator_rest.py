from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def modify_initiator_rest(self, initiator_name, modify_action):
    if self.uuid is None:
        self.module.fail_json(msg='Error modifying igroup initiator %s: igroup not found' % initiator_name)
    api = 'protocols/san/igroups/%s/initiators' % self.uuid
    if modify_action == 'igroup-add':
        body = {'name': initiator_name}
        dummy, error = rest_generic.post_async(self.rest_api, api, body)
    else:
        query = {'allow_delete_while_mapped': self.parameters['force_remove']}
        dummy, error = rest_generic.delete_async(self.rest_api, api, initiator_name.lower(), query)
    if error:
        self.module.fail_json(msg='Error modifying igroup initiator %s: %s' % (initiator_name, error))