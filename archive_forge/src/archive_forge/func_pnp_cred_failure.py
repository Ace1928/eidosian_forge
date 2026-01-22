from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def pnp_cred_failure(self, msg=None):
    """
        Method for failing discovery if there is any discrepancy in the PnP credentials
        passed by the user
        """
    self.log(msg, 'CRITICAL')
    self.module.fail_json(msg=msg)