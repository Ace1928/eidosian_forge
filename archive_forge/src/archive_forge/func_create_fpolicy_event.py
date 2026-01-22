from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
def create_fpolicy_event(self):
    """
        Create an FPolicy policy event
        :return: nothing
        """
    if self.use_rest:
        api = '/protocols/fpolicy/%s/events' % self.vserver_uuid
        body = {'name': self.parameters['name']}
        if 'protocol' in self.parameters:
            body['protocol'] = self.parameters['protocol']
        if 'volume_monitoring' in self.parameters:
            body['volume_monitoring'] = self.parameters['volume_monitoring']
        if 'filters' in self.parameters:
            body['filters'] = self.list_to_dict(self.parameters['filters'])
        if 'file_operations' in self.parameters:
            body['file_operations'] = self.list_to_dict(self.parameters['file_operations'])
        dummy, error = self.rest_api.post(api, body)
        if error:
            self.module.fail_json(msg=error)
    else:
        fpolicy_event_obj = netapp_utils.zapi.NaElement('fpolicy-policy-event-create')
        fpolicy_event_obj.add_new_child('event-name', self.parameters['name'])
        if 'file_operations' in self.parameters:
            file_operation_obj = netapp_utils.zapi.NaElement('file-operations')
            for file_operation in self.parameters['file_operations']:
                file_operation_obj.add_new_child('fpolicy-operation', file_operation)
            fpolicy_event_obj.add_child_elem(file_operation_obj)
        if 'filters' in self.parameters:
            filter_string_obj = netapp_utils.zapi.NaElement('filter-string')
            for filter in self.parameters['filters']:
                filter_string_obj.add_new_child('fpolicy-filter', filter)
            fpolicy_event_obj.add_child_elem(filter_string_obj)
        if 'protocol' in self.parameters:
            fpolicy_event_obj.add_new_child('protocol', self.parameters['protocol'])
        if 'volume_monitoring' in self.parameters:
            fpolicy_event_obj.add_new_child('volume-operation', self.na_helper.get_value_for_bool(from_zapi=False, value=self.parameters['volume_monitoring']))
        try:
            self.server.invoke_successfully(fpolicy_event_obj, True)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error creating fPolicy policy event %s on vserver %s: %s' % (self.parameters['name'], self.parameters['vserver'], to_native(error)), exception=traceback.format_exc())