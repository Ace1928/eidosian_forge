from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def create_broadcast_domain_rest(self, ports=None):
    api = 'network/ethernet/broadcast-domains'
    body = {'name': self.parameters['name'], 'mtu': self.parameters['mtu'], 'ipspace': self.parameters['ipspace']}
    dummy, error = rest_generic.post_async(self.rest_api, api, body)
    if error:
        self.module.fail_json(msg=error)
    if ports:
        self.add_or_move_broadcast_domain_ports_rest(ports)