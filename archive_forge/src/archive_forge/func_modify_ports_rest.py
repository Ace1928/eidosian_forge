from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def modify_ports_rest(self, modify, uuid):
    api = 'network/ethernet/ports'
    body = {}
    if 'ports' in modify:
        member_ports = self.build_member_ports()
        body['lag'] = {'member_ports': member_ports}
    if 'broadcast_domain' in modify or 'ipspace' in modify:
        broadcast_domain = modify['broadcast_domain'] if 'broadcast_domain' in modify else self.parameters['broadcast_domain']
        ipspace = modify['ipspace'] if 'ipspace' in modify else self.parameters['ipspace']
        body['broadcast_domain'] = {'name': broadcast_domain}
        body['broadcast_domain']['ipspace'] = {'name': ipspace}
    dummy, error = rest_generic.patch_async(self.rest_api, api, uuid, body)
    if error:
        self.module.fail_json(msg=error)