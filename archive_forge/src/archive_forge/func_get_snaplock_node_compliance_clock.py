from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
def get_snaplock_node_compliance_clock(self):
    if self.use_rest:
        '\n            Return snaplock-node-compliance-clock query results\n            :return: dict of clock info\n            '
        api = 'private/cli/snaplock/compliance-clock'
        query = {'fields': 'node,time', 'node': self.parameters['node']}
        message, error = self.rest_api.get(api, query)
        records, error = rrh.check_for_0_or_1_records(api, message, error)
        if error is None and records is not None:
            return_value = {'node': message['records'][0]['node'], 'compliance_clock_time': message['records'][0]['time']}
        if error:
            self.module.fail_json(msg=error)
        if not records:
            error = 'REST API did not return snaplock compliance clock for node %s' % self.parameters['node']
            self.module.fail_json(msg=error)
    else:
        '\n            Return snaplock-node-compliance-clock query results\n            :param node_name: name of the cluster node\n            :return: NaElement\n            '
        node_snaplock_clock = netapp_utils.zapi.NaElement('snaplock-get-node-compliance-clock')
        node_snaplock_clock.add_new_child('node', self.parameters['node'])
        try:
            result = self.server.invoke_successfully(node_snaplock_clock, True)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error fetching snaplock compliance clock for node %s : %s' % (self.parameters['node'], to_native(error)), exception=traceback.format_exc())
        return_value = None
        if result.get_child_by_name('snaplock-node-compliance-clock'):
            node_snaplock_clock_attributes = result['snaplock-node-compliance-clock']['compliance-clock-info']
            return_value = {'compliance_clock_time': node_snaplock_clock_attributes['formatted-snaplock-compliance-clock']}
    return return_value