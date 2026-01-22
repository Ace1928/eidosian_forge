from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
def get_fpolicy_scope(self):
    """
        Check to see if the fPolicy scope exists or not
        :return: dict of scope properties if exist, None if not
        """
    return_value = None
    if self.use_rest:
        api = '/private/cli/vserver/fpolicy/policy/scope'
        query = {'vserver': self.parameters['vserver'], 'policy-name': self.parameters['name'], 'fields': 'shares-to-include,shares-to-exclude,volumes-to-include,volumes-to-exclude,export-policies-to-include,export-policies-to-exclude,file-extensions-to-include,file-extensions-to-exclude,is-file-extension-check-on-directories-enabled,is-monitoring-of-objects-with-no-extension-enabled'}
        message, error = self.rest_api.get(api, query)
        records, error = rrh.check_for_0_or_more_records(api, message, error)
        if error:
            self.module.fail_json(msg=error)
        if records is not None:
            return_value = {'name': records[0]['policy_name'], 'check_extensions_on_directories': records[0]['is_file_extension_check_on_directories_enabled'], 'is_monitoring_of_objects_with_no_extension_enabled': records[0]['is_monitoring_of_objects_with_no_extension_enabled']}
            for field in ('export_policies_to_exclude', 'export_policies_to_include', 'export_policies_to_include', 'file_extensions_to_exclude', 'file_extensions_to_include', 'shares_to_exclude', 'shares_to_include', 'volumes_to_exclude', 'volumes_to_include'):
                return_value[field] = []
                if field in records[0]:
                    return_value[field] = records[0][field]
        return return_value
    else:
        fpolicy_scope_obj = netapp_utils.zapi.NaElement('fpolicy-policy-scope-get-iter')
        fpolicy_scope_config = netapp_utils.zapi.NaElement('fpolicy-scope-config')
        fpolicy_scope_config.add_new_child('policy-name', self.parameters['name'])
        fpolicy_scope_config.add_new_child('vserver', self.parameters['vserver'])
        query = netapp_utils.zapi.NaElement('query')
        query.add_child_elem(fpolicy_scope_config)
        fpolicy_scope_obj.add_child_elem(query)
        try:
            result = self.server.invoke_successfully(fpolicy_scope_obj, True)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error searching for FPolicy policy scope %s on vserver %s: %s' % (self.parameters['name'], self.parameters['vserver'], to_native(error)), exception=traceback.format_exc())
        if result.get_child_by_name('attributes-list'):
            fpolicy_scope_attributes = result['attributes-list']['fpolicy-scope-config']
            param_dict = {'export_policies_to_exclude': [], 'export_policies_to_include': [], 'file_extensions_to_exclude': [], 'file_extensions_to_include': [], 'shares_to_exclude': [], 'shares_to_include': [], 'volumes_to_exclude': [], 'volumes_to_include': []}
            for param in param_dict.keys():
                if fpolicy_scope_attributes.get_child_by_name(param.replace('_', '-')):
                    param_dict[param] = [child_name.get_content() for child_name in fpolicy_scope_attributes.get_child_by_name(param.replace('_', '-')).get_children()]
            return_value = {'name': fpolicy_scope_attributes.get_child_content('policy-name'), 'check_extensions_on_directories': self.na_helper.get_value_for_bool(from_zapi=True, value=fpolicy_scope_attributes.get_child_content('check-extensions-on-directories')), 'is_monitoring_of_objects_with_no_extension_enabled': self.na_helper.get_value_for_bool(from_zapi=True, value=fpolicy_scope_attributes.get_child_content('is-monitoring-of-objects-with-no-extension-enabled'))}
            return_value.update(param_dict)
        return return_value