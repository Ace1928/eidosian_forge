from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def get_rule_type(type):
    """Get the rule type of schedule.
        :param type: The schedule type enum
        :return: The rule type of snapshot schedule
    """
    schedule_type = {'ScheduleTypeEnum.N_HOURS_AT_MM': 'every_n_hours', 'ScheduleTypeEnum.DAY_AT_HHMM': 'every_day', 'ScheduleTypeEnum.N_DAYS_AT_HHMM': 'every_n_days', 'ScheduleTypeEnum.SELDAYS_AT_HHMM': 'every_week', 'ScheduleTypeEnum.NTH_DAYOFMONTH_AT_HHMM': 'every_month'}
    return schedule_type.get(type)