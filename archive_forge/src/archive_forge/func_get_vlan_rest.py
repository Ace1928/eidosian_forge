from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_vlan_rest(self):
    api = 'network/ethernet/ports'
    query = {'name': self.parameters['interface_name'], 'node.name': self.parameters['node']}
    fields = 'name,node,uuid,broadcast_domain,enabled'
    record, error = rest_generic.get_one_record(self.rest_api, api, query, fields)
    if error:
        self.module.fail_json(msg=error)
    if record:
        current = {'interface_name': record['name'], 'node': record['node']['name'], 'uuid': record['uuid'], 'enabled': record['enabled']}
        if 'broadcast_domain' in record:
            current['broadcast_domain'] = record['broadcast_domain']['name']
            current['ipspace'] = record['broadcast_domain']['ipspace']['name']
        return current
    return None