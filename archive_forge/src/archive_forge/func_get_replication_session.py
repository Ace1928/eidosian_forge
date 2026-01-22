from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def get_replication_session(self, obj_nas, filter_key=None, replication_params=None, name=None, action=None):
    """ Retrieves the replication sessions configured for the nas server
            :param: obj_nas: NAS server object
            :param: filter_key: Key to filter replication sessions
            :param: replication_params: Module input params
            :param: name: Replication session name
            :param: action: Specifies modify or delete action on replication session
            :return: Replication session details
        """
    try:
        repl_session = self.unity_conn.get_replication_session(src_resource_id=obj_nas.id)
        if not filter_key and repl_session:
            if len(repl_session) > 1:
                if action:
                    error_msg = 'There are multiple replication sessions for the nas server. Please specify replication_name in replication_params to %s.' % action
                    self.module.fail_json(msg=error_msg)
                return repl_session
            return repl_session[0]
        for session in repl_session:
            if filter_key == 'remote_system_name' and session.remote_system.name == replication_params['remote_system_name']:
                return session
            if filter_key == 'name' and session.name == name:
                return session
        return None
    except Exception as e:
        errormsg = ('Retrieving replication session on the nas server failed with error %s', str(e))
        LOG.error(errormsg)
        self.module.fail_json(msg=errormsg)