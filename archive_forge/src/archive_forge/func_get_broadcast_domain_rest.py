from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_broadcast_domain_rest(self, broadcast_domain, ipspace):
    api = 'network/ethernet/broadcast-domains'
    query = {'name': broadcast_domain, 'ipspace.name': ipspace}
    fields = 'uuid,name,ipspace,ports,mtu'
    record, error = rest_generic.get_one_record(self.rest_api, api, query, fields)
    if error:
        self.module.fail_json(msg=error)
    if record:
        current = {'name': record['name'], 'mtu': record['mtu'], 'ipspace': record['ipspace']['name'], 'uuid': record['uuid'], 'ports': []}
        if 'ports' in record:
            current['ports'] = ['%s:%s' % (port['node']['name'], port['name']) for port in record['ports']]
        return current
    return None