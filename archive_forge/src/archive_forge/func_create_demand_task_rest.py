from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def create_demand_task_rest(self):
    api = 'protocols/vscan/%s/on-demand-policies' % self.svm_uuid
    body = {'name': self.parameters['task_name'], 'log_path': self.parameters['report_directory'], 'scan_paths': self.parameters['scan_paths']}
    if self.parameters.get('file_ext_to_exclude'):
        body['scope.exclude_extensions'] = self.parameters['file_ext_to_exclude']
    if self.parameters.get('file_ext_to_include'):
        body['scope.include_extensions'] = self.parameters['file_ext_to_include']
    if self.parameters.get('max_file_size'):
        body['scope.max_file_size'] = self.parameters['max_file_size']
    if self.parameters.get('paths_to_exclude'):
        body['scope.exclude_paths'] = self.parameters['paths_to_exclude']
    if self.parameters.get('scan_files_with_no_ext'):
        body['scope.scan_without_extension'] = self.parameters['scan_files_with_no_ext']
    if self.parameters.get('schedule'):
        body['schedule.name'] = self.parameters['schedule']
    dummy, error = rest_generic.post_async(self.rest_api, api, body)
    if error:
        self.module.fail_json(msg='Error creating on demand task %s: %s' % (self.parameters['task_name'], to_native(error)))