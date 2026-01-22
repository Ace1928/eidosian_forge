from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def get_project_details(self, projectName):
    """
        Get the details of specific project name provided.

        Parameters:
            projectName (str) - Project Name

        Returns:
            items (dict) - Project details with given project name.
        """
    items = self.dnac_apply['exec'](family='configuration_templates', function='get_projects', op_modifies=True, params={'name': projectName})
    return items