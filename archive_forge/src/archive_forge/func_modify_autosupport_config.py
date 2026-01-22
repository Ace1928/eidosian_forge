from __future__ import absolute_import, division, print_function
import re
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def modify_autosupport_config(self, modify):
    """
        modify autosupport config
        @return: modfied attributes / FAILURE with an error_message
        """
    if self.use_rest:
        api = 'private/cli/system/node/autosupport'
        query = {'node': self.parameters['node_name']}
        if 'service_state' in modify:
            modify['state'] = modify['service_state'] == 'started'
            del modify['service_state']
        if 'post_url' in modify:
            modify['url'] = modify.pop('post_url')
        if 'from_address' in modify:
            modify['from'] = modify.pop('from_address')
        if 'to_addresses' in modify:
            modify['to'] = modify.pop('to_addresses')
        if 'hostname_in_subject' in modify:
            modify['hostname_subj'] = modify.pop('hostname_in_subject')
        if 'nht_data_enabled' in modify:
            modify['nht'] = modify.pop('nht_data_enabled')
        if 'perf_data_enabled' in modify:
            modify['perf'] = modify.pop('perf_data_enabled')
        if 'reminder_enabled' in modify:
            modify['reminder'] = modify.pop('reminder_enabled')
        if 'private_data_removed' in modify:
            modify['remove_private_data'] = modify.pop('private_data_removed')
        if 'local_collection_enabled' in modify:
            modify['local_collection'] = modify.pop('local_collection_enabled')
        if 'ondemand_enabled' in modify:
            modify['ondemand_state'] = modify.pop('ondemand_enabled')
        if 'partner_addresses' in modify:
            modify['partner_address'] = modify.pop('partner_addresses')
        dummy, error = rest_generic.patch_async(self.rest_api, api, None, modify, query)
        if error:
            self.module.fail_json(msg='Error modifying asup: %s' % error)
    else:
        asup_details = {'node-name': self.parameters['node_name']}
        if modify.get('service_state'):
            asup_details['is-enabled'] = 'true' if modify.get('service_state') == 'started' else 'false'
        asup_config = netapp_utils.zapi.NaElement('autosupport-config-modify')
        for item_key in modify:
            if item_key in self.na_helper.zapi_string_keys:
                zapi_key = self.na_helper.zapi_string_keys.get(item_key)
                asup_details[zapi_key] = modify[item_key]
            elif item_key in self.na_helper.zapi_int_keys:
                zapi_key = self.na_helper.zapi_int_keys.get(item_key)
                asup_details[zapi_key] = modify[item_key]
            elif item_key in self.na_helper.zapi_bool_keys:
                zapi_key = self.na_helper.zapi_bool_keys.get(item_key)
                asup_details[zapi_key] = self.na_helper.get_value_for_bool(from_zapi=False, value=modify[item_key])
            elif item_key in self.na_helper.zapi_list_keys:
                parent_key, child_key = self.na_helper.zapi_list_keys.get(item_key)
                asup_config.add_child_elem(self.na_helper.get_value_for_list(from_zapi=False, zapi_parent=parent_key, zapi_child=child_key, data=modify.get(item_key)))
        asup_config.translate_struct(asup_details)
        try:
            return self.server.invoke_successfully(asup_config, enable_tunneling=True)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error modifying asup: %s' % to_native(error), exception=traceback.format_exc())