from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def get_project_params(self, params):
    """
        Store project parameters from the playbook for template processing in DNAC.

        Parameters:
            params (dict) - Playbook details containing Project information.

        Returns:
            project_params (dict) - Organized Project parameters.
        """
    project_params = {'name': params.get('project_name'), 'description': params.get('project_description')}
    return project_params