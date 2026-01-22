from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def prepare_output_dict(self, state, share_id, share_name, filesystem_obj, snapshot_obj, nas_server_obj):
    smb_share_details = None
    smb_share_obj = None
    if state == 'present':
        smb_share_obj = self.get_smb_share_obj(share_id, share_name, filesystem_obj, snapshot_obj, nas_server_obj)
        smb_share_details = smb_share_obj._get_properties()
    if smb_share_details:
        if smb_share_obj.type.name == 'CIFS_SNAPSHOT':
            smb_share_details['snapshot_name'] = smb_share_obj.snap.name
            smb_share_details['snapshot_id'] = smb_share_obj.snap.id
        smb_share_details['filesystem_name'] = smb_share_obj.filesystem.name
        smb_share_details['filesystem_id'] = smb_share_obj.filesystem.id
        smb_share_details['nas_server_name'] = smb_share_obj.filesystem.nas_server.name
        smb_share_details['nas_server_id'] = smb_share_obj.filesystem.nas_server.id
    return smb_share_details