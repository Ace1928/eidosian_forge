from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
def create_fpolicy_scope(self):
    """
        Create an FPolicy policy scope
        :return: nothing
        """
    if self.use_rest:
        api = '/private/cli/vserver/fpolicy/policy/scope'
        body = {'vserver': self.parameters['vserver'], 'policy-name': self.parameters['name']}
        for parameter in ('export_policies_to_exclude', 'export_policies_to_include', 'export_policies_to_include', 'file_extensions_to_exclude', 'file_extensions_to_include', 'shares_to_exclude', 'shares_to_include', 'volumes_to_exclude', 'volumes_to_include', 'is-file-extension-check-on-directories-enabled', 'is-monitoring-of-objects-with-no-extension-enabled'):
            if parameter in self.parameters:
                body[parameter.replace('_', '-')] = self.parameters[parameter]
        dummy, error = self.rest_api.post(api, body)
        if error:
            self.module.fail_json(msg=error)
    else:
        fpolicy_scope_obj = netapp_utils.zapi.NaElement('fpolicy-policy-scope-create')
        fpolicy_scope_obj.add_new_child('policy-name', self.parameters['name'])
        if 'check_extensions_on_directories' in self.parameters:
            fpolicy_scope_obj.add_new_child('check-extensions-on-directories', self.na_helper.get_value_for_bool(from_zapi=False, value=self.parameters['check_extensions_on_directories']))
        if 'is_monitoring_of_objects_with_no_extension_enabled' in self.parameters:
            fpolicy_scope_obj.add_new_child('is-monitoring-of-objects-with-no-extension-enabled', self.na_helper.get_value_for_bool(from_zapi=False, value=self.parameters['is_monitoring_of_objects_with_no_extension_enabled']))
        if 'export_policies_to_exclude' in self.parameters:
            export_policies_to_exclude_obj = netapp_utils.zapi.NaElement('export-policies-to-exclude')
            for export_policies_to_exclude in self.parameters['export_policies_to_exclude']:
                export_policies_to_exclude_obj.add_new_child('string', export_policies_to_exclude)
            fpolicy_scope_obj.add_child_elem(export_policies_to_exclude_obj)
        if 'export_policies_to_include' in self.parameters:
            export_policies_to_include_obj = netapp_utils.zapi.NaElement('export-policies-to-include')
            for export_policies_to_include in self.parameters['export_policies_to_include']:
                export_policies_to_include_obj.add_new_child('string', export_policies_to_include)
            fpolicy_scope_obj.add_child_elem(export_policies_to_include_obj)
        if 'file_extensions_to_exclude' in self.parameters:
            file_extensions_to_exclude_obj = netapp_utils.zapi.NaElement('file-extensions-to-exclude')
            for file_extensions_to_exclude in self.parameters['file_extensions_to_exclude']:
                file_extensions_to_exclude_obj.add_new_child('string', file_extensions_to_exclude)
            fpolicy_scope_obj.add_child_elem(file_extensions_to_exclude_obj)
        if 'file_extensions_to_include' in self.parameters:
            file_extensions_to_include_obj = netapp_utils.zapi.NaElement('file-extensions-to-include')
            for file_extensions_to_include in self.parameters['file_extensions_to_include']:
                file_extensions_to_include_obj.add_new_child('string', file_extensions_to_include)
            fpolicy_scope_obj.add_child_elem(file_extensions_to_include_obj)
        if 'shares_to_exclude' in self.parameters:
            shares_to_exclude_obj = netapp_utils.zapi.NaElement('shares-to-exclude')
            for shares_to_exclude in self.parameters['shares_to_exclude']:
                shares_to_exclude_obj.add_new_child('string', shares_to_exclude)
            fpolicy_scope_obj.add_child_elem(shares_to_exclude_obj)
        if 'volumes_to_exclude' in self.parameters:
            volumes_to_exclude_obj = netapp_utils.zapi.NaElement('volumes-to-exclude')
            for volumes_to_exclude in self.parameters['volumes_to_exclude']:
                volumes_to_exclude_obj.add_new_child('string', volumes_to_exclude)
            fpolicy_scope_obj.add_child_elem(volumes_to_exclude_obj)
        if 'shares_to_include' in self.parameters:
            shares_to_include_obj = netapp_utils.zapi.NaElement('shares-to-include')
            for shares_to_include in self.parameters['shares_to_include']:
                shares_to_include_obj.add_new_child('string', shares_to_include)
            fpolicy_scope_obj.add_child_elem(shares_to_include_obj)
        if 'volumes_to_include' in self.parameters:
            volumes_to_include_obj = netapp_utils.zapi.NaElement('volumes-to-include')
            for volumes_to_include in self.parameters['volumes_to_include']:
                volumes_to_include_obj.add_new_child('string', volumes_to_include)
            fpolicy_scope_obj.add_child_elem(volumes_to_include_obj)
        try:
            self.server.invoke_successfully(fpolicy_scope_obj, True)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error creating fPolicy policy scope %s on vserver %s: %s' % (self.parameters['name'], self.parameters['vserver'], to_native(error)), exception=traceback.format_exc())