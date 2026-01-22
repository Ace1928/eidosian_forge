from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
def create_log_forward_config(self):
    """
        Creates a log forward config
        :return: nothing
        """
    if self.use_rest:
        api = 'security/audit/destinations'
        body = dict()
        body['address'] = self.parameters['destination']
        body['port'] = self.parameters['port']
        for attr in ('protocol', 'facility', 'verify_server', 'force'):
            if attr in self.parameters:
                body[attr] = self.parameters[attr]
        dummy, error = self.rest_api.post(api, body)
        if error:
            self.module.fail_json(msg=error)
    else:
        log_forward_config_obj = netapp_utils.zapi.NaElement('cluster-log-forward-create')
        log_forward_config_obj.add_new_child('destination', self.parameters['destination'])
        log_forward_config_obj.add_new_child('port', self.na_helper.get_value_for_int(False, self.parameters['port']))
        if 'facility' in self.parameters:
            log_forward_config_obj.add_new_child('facility', self.parameters['facility'])
        if 'force' in self.parameters:
            log_forward_config_obj.add_new_child('force', self.na_helper.get_value_for_bool(False, self.parameters['force']))
        if 'protocol' in self.parameters:
            log_forward_config_obj.add_new_child('protocol', self.parameters['protocol'])
        if 'verify_server' in self.parameters:
            log_forward_config_obj.add_new_child('verify-server', self.na_helper.get_value_for_bool(False, self.parameters['verify_server']))
        try:
            self.server.invoke_successfully(log_forward_config_obj, True)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error creating log forward config with destination %s on port %s: %s' % (self.parameters['destination'], self.na_helper.get_value_for_int(False, self.parameters['port']), to_native(error)), exception=traceback.format_exc())