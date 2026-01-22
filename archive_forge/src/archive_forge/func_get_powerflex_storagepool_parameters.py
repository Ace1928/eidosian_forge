from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell\
def get_powerflex_storagepool_parameters():
    """This method provides parameters required for the ansible
    Storage Pool module on powerflex"""
    return dict(storage_pool_name=dict(required=False, type='str'), storage_pool_id=dict(required=False, type='str'), protection_domain_name=dict(required=False, type='str'), protection_domain_id=dict(required=False, type='str'), media_type=dict(required=False, type='str', choices=['HDD', 'SSD', 'TRANSITIONAL']), use_rfcache=dict(required=False, type='bool'), use_rmcache=dict(required=False, type='bool'), storage_pool_new_name=dict(required=False, type='str'), state=dict(required=True, type='str', choices=['present', 'absent']))