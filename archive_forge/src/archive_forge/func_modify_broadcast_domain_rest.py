from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def modify_broadcast_domain_rest(self, uuid, modify):
    api = 'network/ethernet/broadcast-domains'
    body = {}
    if 'name' in modify:
        body['name'] = modify['name']
    if 'ipspace' in modify:
        body['ipspace.name'] = modify['ipspace']
    if 'mtu' in modify:
        body['mtu'] = modify['mtu']
    dummy, error = rest_generic.patch_async(self.rest_api, api, uuid, body)
    if error:
        self.module.fail_json(msg=error)