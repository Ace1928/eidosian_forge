from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def get_snapshot_schedules_list(self):
    """ Get the list of snapshot schedules on a given Unity storage
            system """
    try:
        LOG.info('Getting snapshot schedules list ')
        snapshot_schedules = utils.snap_schedule.UnitySnapScheduleList.get(cli=self.unity._cli)
        return result_list(snapshot_schedules)
    except Exception as e:
        msg = 'Get snapshot schedules list from unity array failed with error %s' % str(e)
        LOG.error(msg)
        self.module.fail_json(msg=msg)