from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
def delete_fpolicy_event(self):
    """
        Delete an FPolicy policy event
        :return: nothing
        """
    if self.use_rest:
        api = '/protocols/fpolicy/%s/events/%s' % (self.vserver_uuid, self.parameters['name'])
        dummy, error = self.rest_api.delete(api)
        if error:
            self.module.fail_json(msg=error)
    else:
        fpolicy_event_obj = netapp_utils.zapi.NaElement('fpolicy-policy-event-delete')
        fpolicy_event_obj.add_new_child('event-name', self.parameters['name'])
        try:
            self.server.invoke_successfully(fpolicy_event_obj, True)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error deleting fPolicy policy event %s on vserver %s: %s' % (self.parameters['name'], self.parameters['vserver'], to_native(error)), exception=traceback.format_exc())