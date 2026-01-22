from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def get_pool_drives(self, pool_id=None, pool_name=None):
    """ Get pool drives attached to pool"""
    pool_identifier = pool_id or pool_name
    pool_drives_list = []
    try:
        drive_instances = utils.UnityDiskList.get(self.conn._cli)
        if drive_instances:
            for drive in drive_instances:
                if drive.pool and (drive.pool.id == pool_identifier or drive.pool.name == pool_identifier):
                    pool_drive = {'id': drive.id, 'name': drive.name, 'size': drive.size, 'disk_technology': drive.disk_technology.name, 'tier_type': drive.tier_type.name}
                    pool_drives_list.append(pool_drive)
        LOG.info('Successfully retrieved pool drive details')
        return pool_drives_list
    except Exception as e:
        error_message = 'Get details of pool drives failed with error: {0}'.format(str(e))
        LOG.error(error_message)
        self.module.fail_json(msg=error_message)