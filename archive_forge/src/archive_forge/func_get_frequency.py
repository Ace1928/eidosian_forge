from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.elementsw.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.elementsw.plugins.module_utils.netapp_elementsw_module import NaElementSWModule
def get_frequency(self):
    frequency = None
    if self.schedule_type is not None and self.schedule_type == 'DaysOfWeekFrequency':
        if self.weekdays is not None:
            params = dict(weekdays=self.weekdays)
            if self.days_of_week_hours is not None:
                params['hours'] = self.days_of_week_hours
            if self.days_of_week_minutes is not None:
                params['minutes'] = self.days_of_week_minutes
            frequency = DaysOfWeekFrequency(**params)
    elif self.schedule_type is not None and self.schedule_type == 'DaysOfMonthFrequency':
        if self.days_of_month_monthdays is not None:
            params = dict(monthdays=self.days_of_month_monthdays)
            if self.days_of_month_hours is not None:
                params['hours'] = self.days_of_month_hours
            if self.days_of_month_minutes is not None:
                params['minutes'] = self.days_of_month_minutes
            frequency = DaysOfMonthFrequency(**params)
    elif self.schedule_type is not None and self.schedule_type == 'TimeIntervalFrequency':
        params = dict()
        if self.time_interval_days is not None:
            params['days'] = self.time_interval_days
        if self.time_interval_hours is not None:
            params['hours'] = self.time_interval_hours
        if self.time_interval_minutes is not None:
            params['minutes'] = self.time_interval_minutes
        if not params or sum(params.values()) == 0:
            self.module.fail_json(msg='Specify at least one non zero value with TimeIntervalFrequency.')
        frequency = TimeIntervalFrequency(**params)
    return frequency