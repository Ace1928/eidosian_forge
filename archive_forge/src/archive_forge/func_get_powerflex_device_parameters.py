from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell\
def get_powerflex_device_parameters():
    """This method provide parameter required for the device module on
    PowerFlex"""
    return dict(current_pathname=dict(), device_name=dict(), device_id=dict(), sds_name=dict(), sds_id=dict(), storage_pool_name=dict(), storage_pool_id=dict(), acceleration_pool_id=dict(), acceleration_pool_name=dict(), protection_domain_name=dict(), protection_domain_id=dict(), external_acceleration_type=dict(choices=['Invalid', 'None', 'Read', 'Write', 'ReadAndWrite']), media_type=dict(choices=['HDD', 'SSD', 'NVDIMM']), state=dict(required=True, type='str', choices=['present', 'absent']), force=dict(type='bool', default=False))