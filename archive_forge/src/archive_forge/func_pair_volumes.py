from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.elementsw.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.elementsw.plugins.module_utils.netapp_elementsw_module import NaElementSWModule
from ansible_collections.netapp.elementsw.plugins.module_utils.netapp_module import NetAppModule
def pair_volumes(self):
    """
            Start volume pairing on source, and complete on target volume
        """
    try:
        pair_key = self.elem.start_volume_pairing(volume_id=self.parameters['src_vol_id'], mode=self.parameters['mode'])
        self.dest_elem.complete_volume_pairing(volume_pairing_key=pair_key.volume_pairing_key, volume_id=self.parameters['dest_vol_id'])
    except solidfire.common.ApiServerError as err:
        self.module.fail_json(msg='Error pairing volume id %s' % self.parameters['src_vol_id'], exception=to_native(err))