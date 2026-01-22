from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.elementsw.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.elementsw.plugins.module_utils.netapp_elementsw_module import NaElementSWModule
from ansible_collections.netapp.elementsw.plugins.module_utils.netapp_module import NetAppModule
def pairing_exists(self, src_id, dest_id):
    src_paired = self.check_if_already_paired(self.parameters['src_vol_id'])
    dest_paired = self.check_if_already_paired(self.parameters['dest_vol_id'])
    if src_paired is not None or dest_paired is not None:
        return True
    return None