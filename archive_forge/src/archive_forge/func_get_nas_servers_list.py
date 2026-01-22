from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def get_nas_servers_list(self):
    """Get the list of NAS servers on a given Unity storage system"""
    try:
        LOG.info('Getting NAS servers list')
        nas_servers = self.unity.get_nas_server()
        return result_list(nas_servers)
    except Exception as e:
        msg = 'Get NAS servers list from unity array failed with error %s' % str(e)
        LOG.error(msg)
        self.module.fail_json(msg=msg)