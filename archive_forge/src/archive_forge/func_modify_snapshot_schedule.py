from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def modify_snapshot_schedule(self, id, schedule_details):
    """Modify snapshot schedule details.
            :param id: The id of the snapshot schedule
            :param schedule_details: The dict containing schedule details
            :return: The boolean value to indicate if snapshot schedule
             modified
        """
    try:
        obj_schedule = self.return_schedule_instance(id=id)
        rule_id = schedule_details['rules'][0]['id']
        if self.module.params['auto_delete'] is None:
            auto_delete = schedule_details['rules'][0]['is_auto_delete']
        else:
            auto_delete = self.module.params['auto_delete']
        if schedule_details['rules'][0]['is_auto_delete'] and self.module.params['desired_retention'] and (self.module.params['auto_delete'] is False):
            auto_delete = False
        elif schedule_details['rules'][0]['retention_time']:
            auto_delete = None
        rule_dict = self.create_rule(self.module.params['type'], self.module.params['interval'], self.module.params['hours_of_day'], self.module.params['day_interval'], self.module.params['days_of_week'], self.module.params['day_of_month'], self.module.params['hour'], self.module.params['minute'], self.module.params['desired_retention'], self.module.params['retention_unit'], auto_delete, schedule_details)
        obj_schedule.modify(add_rules=[rule_dict], remove_rule_ids=[rule_id])
        return True
    except Exception as e:
        errormsg = 'Modify operation of snapshot schedule id:{0} failed with error {1}'.format(id, str(e))
        LOG.error(errormsg)
        self.module.fail_json(msg=errormsg)