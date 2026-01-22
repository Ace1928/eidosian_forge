from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def get_disk_groups_list(self):
    """Get the list of disk group details on a given Unity storage system"""
    try:
        LOG.info('Getting disk group list')
        pool_disk_list = []
        disk_instances = utils.UnityDiskGroupList(cli=self.unity._cli)
        if disk_instances:
            for disk in disk_instances:
                pool_disk = {'id': disk.id, 'name': disk.name, 'tier_type': disk.tier_type.name}
                pool_disk_list.append(pool_disk)
        return pool_disk_list
    except Exception as e:
        msg = 'Get disk group from unity array failed with error %s' % str(e)
        LOG.error(msg)
        self.module.fail_json(msg=msg)