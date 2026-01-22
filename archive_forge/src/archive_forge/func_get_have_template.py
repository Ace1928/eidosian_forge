from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def get_have_template(self, config, template_available):
    """
        Get the current template related information from DNAC.

        Parameters:
            config (dict) - Playbook details containing Template information.
            template_available (list) -  Current project information.

        Returns:
            self
        """
    projectName = config.get('configuration_templates').get('project_name')
    templateName = config.get('configuration_templates').get('template_name')
    template = None
    have_template = {}
    have_template['isCommitPending'] = False
    have_template['template_found'] = False
    template_details = get_dict_result(template_available, 'name', templateName)
    if not template_details:
        self.log('Template {0} not found in project {1}'.format(templateName, projectName), 'INFO')
        self.msg = 'Template : {0} missing, new template to be created'.format(templateName)
        self.status = 'success'
        return self
    config['templateId'] = template_details.get('id')
    have_template['id'] = template_details.get('id')
    template_list = self.dnac_apply['exec'](family='configuration_templates', function='gets_the_templates_available', params={'projectNames': config.get('projectName')})
    have_template['isCommitPending'] = True
    if template_list and isinstance(template_list, list):
        template_info = get_dict_result(template_list, 'name', templateName)
        if template_info:
            template = self.get_template(config)
            have_template['template'] = template
            have_template['isCommitPending'] = False
            have_template['template_found'] = template is not None and isinstance(template, dict)
            self.log('Template {0} is found and template details are :{1}'.format(templateName, str(template)), 'INFO')
    self.log('Commit pending for template name {0} is {1}'.format(templateName, have_template.get('isCommitPending')), 'INFO')
    self.have_template = have_template
    self.msg = 'Successfully collected all template parameters from dnac for comparison'
    self.status = 'success'
    return self