from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def return_schedule_instance(self, id):
    """Return the snapshot schedule instance
            :param id: The id of the snapshot schedule
            :return: Instance of the snapshot schedule
        """
    try:
        obj_schedule = utils.snap_schedule.UnitySnapSchedule.get(self.unity_conn._cli, id)
        return obj_schedule
    except Exception as e:
        error_msg = 'Failed to get the snapshot schedule {0} instance with error {1}'.format(id, str(e))
        LOG.error(error_msg)
        self.module.fail_json(msg=error_msg)