from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.elementsw.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.elementsw.plugins.module_utils.netapp_elementsw_module import NaElementSWModule
def get_volume_ids(self):
    volume_ids = []
    for volume in self.volumes:
        volume_id = self.elementsw_helper.volume_exists(volume, self.account_id)
        if volume_id:
            volume_ids.append(volume_id)
        else:
            self.module.fail_json(msg='Error: Specified volume %s does not exist' % volume)
    return volume_ids