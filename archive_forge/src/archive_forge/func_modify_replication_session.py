from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def modify_replication_session(self, nas_server_obj, repl_session, replication_params):
    """ Modify the replication session
            :param: nas_server_obj: NAS server object
            :param: repl_session: Replication session to be modified
            :param: replication_params: Module input params
            :return: True if modification is successful
        """
    try:
        LOG.info('Modifying replication session of nas server %s', nas_server_obj.name)
        modify_payload = {}
        if replication_params['replication_mode'] and replication_params['replication_mode'] == 'manual':
            rpo = -1
        elif replication_params['rpo']:
            rpo = replication_params['rpo']
        name = repl_session.name
        if replication_params['new_replication_name'] and name != replication_params['new_replication_name']:
            name = replication_params['new_replication_name']
        if repl_session.name != name:
            modify_payload['name'] = name
        if (replication_params['replication_mode'] or replication_params['rpo']) and repl_session.max_time_out_of_sync != rpo:
            modify_payload['max_time_out_of_sync'] = rpo
        if modify_payload:
            repl_session.modify(**modify_payload)
            return True
        return False
    except Exception as e:
        errormsg = ('Modifying replication session failed with error %s', e)
        LOG.error(errormsg)
        self.module.fail_json(msg=errormsg)