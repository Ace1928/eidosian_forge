from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.elementsw.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.elementsw.plugins.module_utils.netapp_elementsw_module import NaElementSWModule
def delete_access_group(self):
    """
        Delete the Access Group
        """
    try:
        self.sfe.delete_volume_access_group(volume_access_group_id=self.group_id)
    except Exception as e:
        self.module.fail_json(msg='Error deleting volume access group %s: %s' % (self.access_group_name, to_native(e)), exception=traceback.format_exc())