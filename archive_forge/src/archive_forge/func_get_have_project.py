from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def get_have_project(self, config):
    """
        Get the current project related information from DNAC.

        Parameters:
            config (dict) - Playbook details containing Project information.

        Returns:
            template_available (list) - Current project information.
        """
    have_project = {}
    given_projectName = config.get('configuration_templates').get('project_name')
    template_available = None
    project_details = self.get_project_details(given_projectName)
    if not (project_details and isinstance(project_details, list)):
        self.log('Project: {0} not found, need to create new project in DNAC'.format(given_projectName), 'INFO')
        return None
    fetched_projectName = project_details[0].get('name')
    if fetched_projectName != given_projectName:
        self.log('Project {0} provided is not exact match in DNAC DB'.format(given_projectName), 'INFO')
        return None
    template_available = project_details[0].get('templates')
    have_project['project_found'] = True
    have_project['id'] = project_details[0].get('id')
    have_project['isDeletable'] = project_details[0].get('isDeletable')
    self.have_project = have_project
    return template_available