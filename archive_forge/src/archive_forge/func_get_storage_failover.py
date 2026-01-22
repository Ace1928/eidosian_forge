from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_storage_failover(self):
    """
        get the storage failover for a given node
        :return: dict of is-enabled: true if enabled is true None if not
        """
    if self.use_rest:
        return_value = None
        api = 'cluster/nodes'
        query = {'fields': 'uuid,ha.enabled', 'name': self.parameters['node_name']}
        record, error = rest_generic.get_one_record(self.rest_api, api, query)
        if error:
            self.module.fail_json(msg=error)
        if not record:
            msg = self.get_node_names_as_str()
            error = 'REST API did not return failover details for node %s, %s' % (self.parameters['node_name'], msg)
            self.module.fail_json(msg=error)
        return_value = {'uuid': record['uuid']}
        if 'ha' in record:
            return_value['is_enabled'] = record['ha']['enabled']
    else:
        storage_failover_get_iter = netapp_utils.zapi.NaElement('cf-status')
        storage_failover_get_iter.add_new_child('node', self.parameters['node_name'])
        try:
            result = self.server.invoke_successfully(storage_failover_get_iter, True)
            return_value = {'is_enabled': self.na_helper.get_value_for_bool(True, result.get_child_content('is-enabled'))}
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error getting storage failover info for node %s: %s' % (self.parameters['node_name'], to_native(error)), exception=traceback.format_exc())
    return return_value