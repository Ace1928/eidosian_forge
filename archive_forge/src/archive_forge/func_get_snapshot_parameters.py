from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
from datetime import datetime
def get_snapshot_parameters():
    """This method provide parameter required for the ansible filesystem
    snapshot module on Unity"""
    return dict(snapshot_name=dict(required=False, type='str'), snapshot_id=dict(required=False, type='str'), filesystem_name=dict(required=False, type='str'), filesystem_id=dict(required=False, type='str'), nas_server_name=dict(required=False, type='str'), nas_server_id=dict(required=False, type='str'), auto_delete=dict(required=False, type='bool'), expiry_time=dict(required=False, type='str'), description=dict(required=False, type='str'), fs_access_type=dict(required=False, type='str', choices=['Checkpoint', 'Protocol']), state=dict(required=True, type='str', choices=['present', 'absent']))