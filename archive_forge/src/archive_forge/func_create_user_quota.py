from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def create_user_quota(self, fs_id, soft_limit, hard_limit, unit, uid, unix, win_name, tree_quota_id):
    """
            Create user quota of a filesystem.
            :param fs_id: ID of filesystem where user quota is to be created.
            :param soft_limit: Soft limit
            :param hard_limit: Hard limit
            :param unit: Unit of soft limit and hard limit
            :param uid: UID of the user quota
            :param unix: Unix user name of user quota
            :param win_name: Windows user name of user quota
            :param tree_quota_id: ID of tree quota
            :return: Object containing new user quota details.
        """
    unix_or_uid_or_win = uid if uid else unix if unix else win_name
    fs_id_or_tree_quota_id = fs_id if fs_id else tree_quota_id
    if soft_limit is None and hard_limit is None:
        errormsg = 'Both soft limit and hard limit cannot be empty. Please provide atleast one to create user quota.'
        LOG.error(errormsg)
        self.module.fail_json(msg=errormsg)
    soft_limit_in_bytes = utils.get_size_bytes(soft_limit, unit)
    hard_limit_in_bytes = utils.get_size_bytes(hard_limit, unit)
    try:
        if self.check_user_type_provided(win_name, uid, unix):
            obj_user_quota = self.unity_conn.create_user_quota(filesystem_id=fs_id, hard_limit=hard_limit_in_bytes, soft_limit=soft_limit_in_bytes, uid=uid, unix_name=unix, win_name=win_name, tree_quota_id=tree_quota_id)
            LOG.info('Successfully created user quota')
            return obj_user_quota
    except Exception as e:
        errormsg = 'Create quota for user {0} on {1} , failed with error {2} '.format(unix_or_uid_or_win, fs_id_or_tree_quota_id, str(e))
        LOG.error(errormsg)
        self.module.fail_json(msg=errormsg)