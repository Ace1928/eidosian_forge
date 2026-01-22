from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def modify_filesystem(self, update_dict, obj_fs):
    """ modifes attributes for a filesystem instance
        :param update_dict: modify dict
        :return: True on Success
        """
    try:
        adv_smb_params = ['is_cifs_sync_writes_enabled', 'is_cifs_op_locks_enabled', 'is_cifs_notify_on_write_enabled', 'is_cifs_notify_on_access_enabled', 'cifs_notify_on_change_dir_depth']
        cifs_fs_payload = {}
        fs_update_payload = {}
        for smb_param in adv_smb_params:
            if smb_param in update_dict.keys():
                cifs_fs_payload.update({smb_param: update_dict[smb_param]})
        LOG.debug('CIFS Modify Payload: %s', cifs_fs_payload)
        cifs_fs_parameters = obj_fs.prepare_cifs_fs_parameters(**cifs_fs_payload)
        fs_update_params = ['size', 'is_thin', 'tiering_policy', 'is_compression', 'access_policy', 'locking_policy', 'description', 'cifs_fs_parameters']
        for fs_param in fs_update_params:
            if fs_param in update_dict.keys():
                fs_update_payload.update({fs_param: update_dict[fs_param]})
        if cifs_fs_parameters:
            fs_update_payload.update({'cifs_fs_parameters': cifs_fs_parameters})
        if 'snap_sch_id' in update_dict.keys():
            fs_update_payload.update({'snap_schedule_parameters': {'snapSchedule': {'id': update_dict.get('snap_sch_id')}}})
        elif 'is_snap_schedule_paused' in update_dict.keys():
            fs_update_payload.update({'snap_schedule_parameters': {'isSnapSchedulePaused': False}})
        obj_fs = obj_fs.update()
        resp = obj_fs.modify(**fs_update_payload)
        LOG.info('Successfully modified the FS with response %s', resp)
    except Exception as e:
        errormsg = 'Failed to modify FileSystem instance id: {0} with error {1}'.format(obj_fs.id, str(e))
        LOG.error(errormsg)
        self.module.fail_json(msg=errormsg)