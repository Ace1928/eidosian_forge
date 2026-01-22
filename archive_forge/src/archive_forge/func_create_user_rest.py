from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
def create_user_rest(self, apps):
    api = 'security/accounts'
    body = {'name': self.parameters['name'], 'role.name': self.parameters['role_name'], 'applications': self.na_helper.filter_out_none_entries(apps)}
    if self.parameters.get('vserver') is not None:
        body['owner.name'] = self.parameters['vserver']
    if 'set_password' in self.parameters:
        body['password'] = self.parameters['set_password']
    if 'lock_user' in self.parameters:
        body['locked'] = self.parameters['lock_user']
    dummy, error = self.rest_api.post(api, body)
    if error and 'invalid value' in error['message'] and any((x in error['message'] for x in ['service-processor', 'service_processor'])):
        app_list_sp = body['applications']
        for app_item in app_list_sp:
            if app_item['application'] == 'service-processor':
                app_item['application'] = 'service_processor'
            elif app_item['application'] == 'service_processor':
                app_item['application'] = 'service-processor'
        body['applications'] = app_list_sp
        dummy, error_sp = self.rest_api.post(api, body)
        if not error_sp:
            return
    if error:
        self.module.fail_json(msg='Error while creating user: %s' % error)