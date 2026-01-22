from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.elementsw.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.elementsw.plugins.module_utils.netapp_elementsw_module import NaElementSWModule
def rename_account(self):
    """
        Rename the Account
        """
    try:
        self.sfe.modify_account(account_id=self.account_id, username=self.element_username, status=self.status, initiator_secret=self.initiator_secret, target_secret=self.target_secret, attributes=self.attributes)
    except Exception as e:
        self.module.fail_json(msg='Error renaming account %s: %s' % (self.account_id, to_native(e)), exception=traceback.format_exc())