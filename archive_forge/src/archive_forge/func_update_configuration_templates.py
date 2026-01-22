from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def update_configuration_templates(self, config):
    """
        Update/Create templates and projects in DNAC with fields provided in DNAC.

        Parameters:
            config (dict) - Playbook details containing template information.

        Returns:
            self
        """
    configuration_templates = config.get('configuration_templates')
    if configuration_templates:
        is_project_found = self.have_project.get('project_found')
        if not is_project_found:
            project_id, project_created = self.create_project_or_template(is_create_project=True)
            if project_created:
                self.log('project created with projectId: {0}'.format(project_id), 'DEBUG')
            else:
                self.status = 'failed'
                self.msg = 'Project creation failed'
                return self
        is_template_found = self.have_template.get('template_found')
        template_params = self.want.get('template_params')
        self.log('Desired template details: {0}'.format(template_params), 'DEBUG')
        self.log('Current template details: {0}'.format(self.have_template), 'DEBUG')
        template_id = None
        template_updated = False
        self.validate_input_merge(is_template_found).check_return_status()
        if is_template_found:
            if self.requires_update():
                template_id = self.have_template.get('id')
                template_params.update({'id': template_id})
                self.log('Current State (have): {0}'.format(self.have_template), 'INFO')
                self.log('Desired State (want): {0}'.format(self.want), 'INFO')
                response = self.dnac_apply['exec'](family='configuration_templates', function='update_template', params=template_params, op_modifies=True)
                template_updated = True
                self.log("Updating existing template '{0}'.".format(self.have_template.get('template').get('name')), 'INFO')
            else:
                self.result.update({'response': self.have_template.get('template'), 'msg': 'Template does not need update'})
                self.status = 'exited'
                return self
        elif template_params.get('name'):
            template_id, template_updated = self.create_project_or_template()
        else:
            self.msg = 'missing required arguments: template_name'
            self.status = 'failed'
            return self
        if template_updated:
            version_params = {'comments': self.want.get('comments'), 'templateId': template_id}
            response = self.dnac_apply['exec'](family='configuration_templates', function='version_template', op_modifies=True, params=version_params)
            task_id = response.get('response').get('taskId')
            if not task_id:
                self.msg = 'Task id: {0} not found'.format(task_id)
                self.status = 'failed'
                return self
            task_details = self.get_task_details(task_id)
            self.result['changed'] = True
            self.result['msg'] = task_details.get('progress')
            self.result['diff'] = config.get('configuration_templates')
            self.log("Task details for 'version_template': {0}".format(task_details), 'DEBUG')
            self.result['response'] = task_details if task_details else response
            if not self.result.get('msg'):
                self.msg = 'Error while versioning the template'
                self.status = 'failed'
                return self