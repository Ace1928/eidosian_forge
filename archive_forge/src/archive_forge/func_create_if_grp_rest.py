from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def create_if_grp_rest(self):
    api = 'network/ethernet/ports'
    body = {'type': 'lag', 'node': {'name': self.parameters['node']}, 'lag': {'mode': self.parameters['mode'], 'distribution_policy': self.parameters['distribution_function']}}
    if self.parameters.get('ports') is not None:
        body['lag']['member_ports'] = self.build_member_ports()
    if 'broadcast_domain' in self.parameters:
        body['broadcast_domain'] = {'name': self.parameters['broadcast_domain']}
        body['broadcast_domain']['ipspace'] = {'name': self.parameters['ipspace']}
    dummy, error = rest_generic.post_async(self.rest_api, api, body)
    if error:
        self.module.fail_json(msg=error)