from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_lun_map_reporting_nodes_rest(self):
    api = 'protocols/san/lun-maps'
    query = {'lun.name': self.parameters['path'], 'igroup.name': self.parameters['initiator_group_name'], 'svm.name': self.parameters['vserver'], 'fields': 'reporting_nodes,lun.uuid,igroup.uuid'}
    record, error = rest_generic.get_one_record(self.rest_api, api, query)
    if error:
        self.module.fail_json(msg='Error getting LUN map for %s: %s' % (self.parameters['initiator_group_name'], to_native(error)))
    if record:
        self.lun_uuid = record['lun']['uuid']
        self.igroup_uuid = record['igroup']['uuid']
        node_list = []
        for node in record.get('reporting_nodes', []):
            self.nodes_uuids[node['name']] = node['uuid']
            node_list.append(node['name'])
        return node_list
    return None