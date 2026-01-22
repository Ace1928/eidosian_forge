from __future__ import absolute_import, division, print_function
import copy
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_volume_efficiency(self):
    """
        get the storage efficiency for a given path
        :return: dict of sis if exist, None if not
        """
    return_value = None
    if self.use_rest:
        api = 'storage/volumes'
        query = {'svm.name': self.parameters['vserver'], 'fields': 'uuid,efficiency'}
        if self.parameters.get('path'):
            query['efficiency.volume_path'] = self.parameters['path']
        else:
            query['name'] = self.parameters['volume_name']
        record, error = rest_generic.get_one_record(self.rest_api, api, query)
        if error:
            path_or_volume = self.parameters.get('path') or self.parameters.get('volume_name')
            self.module.fail_json(msg='Error getting volume efficiency for path %s on vserver %s: %s' % (path_or_volume, self.parameters['vserver'], to_native(error)), exception=traceback.format_exc())
        if record:
            return_value = self.format_rest_record(record)
        return return_value
    else:
        sis_get_iter = netapp_utils.zapi.NaElement('sis-get-iter')
        sis_status_info = netapp_utils.zapi.NaElement('sis-status-info')
        sis_status_info.add_new_child('path', self.parameters['path'])
        query = netapp_utils.zapi.NaElement('query')
        query.add_child_elem(sis_status_info)
        sis_get_iter.add_child_elem(query)
        try:
            result = self.server.invoke_successfully(sis_get_iter, True)
            if result.get_child_by_name('attributes-list'):
                sis_status_attributes = result['attributes-list']['sis-status-info']
                return_value = {'path': sis_status_attributes['path'], 'enabled': sis_status_attributes['state'], 'status': sis_status_attributes['status'], 'schedule': sis_status_attributes['schedule'], 'enable_inline_compression': self.na_helper.get_value_for_bool(True, sis_status_attributes.get_child_content('is-inline-compression-enabled')), 'enable_compression': self.na_helper.get_value_for_bool(True, sis_status_attributes.get_child_content('is-compression-enabled')), 'enable_inline_dedupe': self.na_helper.get_value_for_bool(True, sis_status_attributes.get_child_content('is-inline-dedupe-enabled')), 'enable_data_compaction': self.na_helper.get_value_for_bool(True, sis_status_attributes.get_child_content('is-data-compaction-enabled')), 'enable_cross_volume_inline_dedupe': self.na_helper.get_value_for_bool(True, sis_status_attributes.get_child_content('is-cross-volume-inline-dedupe-enabled')), 'enable_cross_volume_background_dedupe': self.na_helper.get_value_for_bool(True, sis_status_attributes.get_child_content('is-cross-volume-background-dedupe-enabled'))}
                if sis_status_attributes.get_child_by_name('policy'):
                    return_value['policy'] = sis_status_attributes['policy']
                else:
                    return_value['policy'] = '-'
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error getting volume efficiency for path %s on vserver %s: %s' % (self.parameters['path'], self.parameters['vserver'], to_native(error)), exception=traceback.format_exc())
        return return_value