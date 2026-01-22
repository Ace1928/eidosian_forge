from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def create_demand_task(self):
    """
        Create a Demand Task
        :return: None
        """
    if self.use_rest:
        return self.create_demand_task_rest()
    demand_task_obj = netapp_utils.zapi.NaElement('vscan-on-demand-task-create')
    demand_task_obj.add_new_child('report-directory', self.parameters['report_directory'])
    demand_task_obj.add_new_child('task-name', self.parameters['task_name'])
    scan_paths = netapp_utils.zapi.NaElement('scan-paths')
    for scan_path in self.parameters['scan_paths']:
        scan_paths.add_new_child('string', scan_path)
    demand_task_obj.add_child_elem(scan_paths)
    if self.parameters.get('cross_junction'):
        demand_task_obj.add_new_child('cross-junction', str(self.parameters['cross_junction']).lower())
    if self.parameters.get('directory_recursion'):
        demand_task_obj.add_new_child('directory-recursion', str(self.parameters['directory_recursion']).lower())
    if self.parameters.get('file_ext_to_exclude'):
        ext_to_exclude_obj = netapp_utils.zapi.NaElement('file-ext-to-exclude')
        for exclude_file in self.parameters['file_ext_to_exclude']:
            ext_to_exclude_obj.add_new_child('file-extension', exclude_file)
        demand_task_obj.add_child_elem(ext_to_exclude_obj)
    if self.parameters.get('file_ext_to_include'):
        ext_to_include_obj = netapp_utils.zapi.NaElement('file-ext-to-include')
        for include_file in self.parameters['file_ext_to_exclude']:
            ext_to_include_obj.add_child_elem(include_file)
        demand_task_obj.add_child_elem(ext_to_include_obj)
    if self.parameters.get('max_file_size'):
        demand_task_obj.add_new_child('max-file-size', str(self.parameters['max_file_size']))
    if self.parameters.get('paths_to_exclude'):
        exclude_paths = netapp_utils.zapi.NaElement('paths-to-exclude')
        for path in self.parameters['paths_to_exclude']:
            exclude_paths.add_new_child('string', path)
        demand_task_obj.add_child_elem(exclude_paths)
    if self.parameters.get('report_log_level'):
        demand_task_obj.add_new_child('report-log-level', self.parameters['report_log_level'])
    if self.parameters.get('request_timeout'):
        demand_task_obj.add_new_child('request-timeout', self.parameters['request_timeout'])
    if self.parameters.get('scan_files_with_no_ext'):
        demand_task_obj.add_new_child('scan-files-with-no-ext', str(self.parameters['scan_files_with_no_ext']).lower())
    if self.parameters.get('scan_priority'):
        demand_task_obj.add_new_child('scan-priority', self.parameters['scan_priority'].lower())
    if self.parameters.get('schedule'):
        demand_task_obj.add_new_child('schedule', self.parameters['schedule'])
    try:
        self.server.invoke_successfully(demand_task_obj, True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error creating on demand task %s: %s' % (self.parameters['task_name'], to_native(error)), exception=traceback.format_exc())