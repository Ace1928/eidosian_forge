from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def create_project_or_template(self, is_create_project=False):
    """
        Call DNAC API to create project or template based on the input provided.

        Parameters:
            is_create_project (bool) - Default value is False.

        Returns:
            creation_id (str) - Project Id.
            created (str) - True if Project created, else False.
        """
    creation_id = None
    created = False
    self.log('Desired State (want): {0}'.format(self.want), 'INFO')
    template_params = self.want.get('template_params')
    project_params = self.want.get('project_params')
    if is_create_project:
        params_key = project_params
        name = 'project: {0}'.format(project_params.get('name'))
        validation_string = 'Successfully created project'
        creation_value = 'create_project'
    else:
        params_key = template_params
        name = 'template: {0}'.format(template_params.get('name'))
        validation_string = 'Successfully created template'
        creation_value = 'create_template'
    response = self.dnac_apply['exec'](family='configuration_templates', function=creation_value, op_modifies=True, params=params_key)
    if not isinstance(response, dict):
        self.log("Response of '{0}' is not in dictionary format.".format(creation_value), 'CRITICAL')
        return (creation_id, created)
    task_id = response.get('response').get('taskId')
    if not task_id:
        self.log("Task id {0} not found for '{1}'.".format(task_id, creation_value), 'CRITICAL')
        return (creation_id, created)
    while not created:
        task_details = self.get_task_details(task_id)
        if not task_details:
            self.log("Failed to get task details of '{0}' for taskid: {1}".format(creation_value, task_id), 'CRITICAL')
            return (creation_id, created)
        self.log('Task details for {0}: {1}'.format(creation_value, task_details), 'DEBUG')
        if task_details.get('isError'):
            self.log("Error occurred for '{0}' with taskid: {1}".format(creation_value, task_id), 'ERROR')
            return (creation_id, created)
        if validation_string not in task_details.get('progress'):
            self.log("'{0}' progress set to {1} for taskid: {2}".format(creation_value, task_details.get('progress'), task_id), 'DEBUG')
            continue
        task_details_data = task_details.get('data')
        value = self.check_string_dictionary(task_details_data)
        if value is None:
            creation_id = task_details.get('data')
        else:
            creation_id = value.get('templateId')
        if not creation_id:
            self.log("Export data is not found for '{0}' with taskid : {1}".format(creation_value, task_id), 'DEBUG')
            continue
        created = True
        if is_create_project:
            template_params['projectId'] = creation_id
            template_params['project_id'] = creation_id
    self.log('New {0} created with id {1}'.format(name, creation_id), 'DEBUG')
    return (creation_id, created)