from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def get_changed(self):
    """
        Returns if anything changed
        Args: none

        Returns: boolean
        """
    return self.__changed