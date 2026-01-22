from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_subsystem_host_map_rest(self, type):
    if type == 'hosts':
        api = 'protocols/nvme/subsystems/%s/hosts' % self.subsystem_uuid
        records, error = rest_generic.get_0_or_more_records(self.rest_api, api)
        if error:
            self.module.fail_json(msg='Error fetching subsystem host info for vserver: %s: %s' % (self.parameters['vserver'], to_native(error)))
        if records is not None:
            return {type: [record['nqn'] for record in records]}
        return None
    if type == 'paths':
        api = 'protocols/nvme/subsystem-maps'
        query = {'svm.name': self.parameters['vserver'], 'subsystem.name': self.parameters['subsystem']}
        records, error = rest_generic.get_0_or_more_records(self.rest_api, api, query)
        if error:
            self.module.fail_json(msg='Error fetching subsystem map info for vserver: %s: %s' % (self.parameters['vserver'], to_native(error)))
        if records is not None:
            return_list = []
            for each in records:
                return_list.append(each['namespace']['name'])
                self.namespace_list.append(each['namespace'])
            return {type: return_list}
        return None