from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def get_info_parameters():
    """This method provides parameters required for the ansible
    info module on Unity"""
    return dict(gather_subset=dict(type='list', required=False, elements='str', choices=['host', 'fc_initiator', 'iscsi_initiator', 'cg', 'storage_pool', 'vol', 'snapshot_schedule', 'nas_server', 'file_system', 'snapshot', 'nfs_export', 'smb_share', 'user_quota', 'tree_quota', 'disk_group', 'nfs_server', 'cifs_server', 'ethernet_port', 'file_interface', 'replication_session']))