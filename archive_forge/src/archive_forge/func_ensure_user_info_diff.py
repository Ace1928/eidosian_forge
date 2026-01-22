from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_text
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec
def ensure_user_info_diff(self, user_account):
    """
        Ensure a user information difference.
        The method can check a user description difference only.
        Also, it can't get the set password via vSphere API.

        Returns: bool
        """
    if user_account.fullName != self.user_description and self.user_description is not None:
        return True
    return False