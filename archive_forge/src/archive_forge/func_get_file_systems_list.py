from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def get_file_systems_list(self):
    """Get the list of file systems on a given Unity storage system"""
    try:
        LOG.info('Getting file systems list')
        file_systems = self.unity.get_filesystem()
        return result_list(file_systems)
    except Exception as e:
        msg = 'Get file systems list from unity array failed with error %s' % str(e)
        LOG.error(msg)
        self.module.fail_json(msg=msg)