from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.elementsw.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.elementsw.plugins.module_utils.netapp_elementsw_module import NaElementSWModule
def does_admin_user_exist(self):
    """
        Checks to see if an admin user exists or not
        :return: True if the user exist, False if it dose not exist
        """
    admins_list = self.sfe.list_cluster_admins()
    for admin in admins_list.cluster_admins:
        if admin.username == self.element_username:
            return True
    return False