from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def get_template_params(self, params):
    """
        Store template parameters from the playbook for template processing in DNAC.

        Parameters:
            params (dict) - Playbook details containing Template information.

        Returns:
            temp_params (dict) - Organized template parameters.
        """
    self.log('Template params playbook details: {0}'.format(params), 'DEBUG')
    temp_params = {'tags': self.get_tags(params.get('template_tag')), 'author': params.get('author'), 'composite': params.get('composite'), 'containingTemplates': self.get_containing_templates(params.get('containing_templates')), 'createTime': params.get('create_time'), 'customParamsOrder': params.get('custom_params_order'), 'description': params.get('template_description'), 'deviceTypes': self.get_device_types(params.get('device_types')), 'failurePolicy': params.get('failure_policy'), 'id': params.get('id'), 'language': params.get('language').upper(), 'lastUpdateTime': params.get('last_update_time'), 'latestVersionTime': params.get('latest_version_time'), 'name': params.get('template_name'), 'parentTemplateId': params.get('parent_template_id'), 'projectId': params.get('project_id'), 'projectName': params.get('project_name'), 'rollbackTemplateContent': params.get('rollback_template_content'), 'rollbackTemplateParams': self.get_template_info(params.get('rollback_template_params')), 'softwareType': params.get('software_type'), 'softwareVariant': params.get('software_variant'), 'softwareVersion': params.get('software_version'), 'templateContent': params.get('template_content'), 'templateParams': self.get_template_info(params.get('template_params')), 'validationErrors': self.get_validation_errors(params.get('validation_errors')), 'version': params.get('version'), 'project_id': params.get('project_id')}
    self.log('Formatted template params details: {0}'.format(temp_params), 'DEBUG')
    copy_temp_params = copy.deepcopy(temp_params)
    for item in copy_temp_params:
        if temp_params[item] is None:
            del temp_params[item]
    self.log('Formatted template params details: {0}'.format(temp_params), 'DEBUG')
    return temp_params