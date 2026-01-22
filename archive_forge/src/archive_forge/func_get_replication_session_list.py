from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def get_replication_session_list(self):
    """Get the list of replication sessions on a given Unity storage system"""
    try:
        LOG.info('Getting replication sessions list')
        replication_sessions = self.unity.get_replication_session()
        return result_list(replication_sessions)
    except Exception as e:
        msg = 'Get replication session list from unity array failed with error %s' % str(e)
        LOG.error(msg)
        self.module.fail_json(msg=msg)