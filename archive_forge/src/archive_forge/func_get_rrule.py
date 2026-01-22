from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import raise_from
from ansible.plugins.lookup import LookupBase
from ansible.errors import AnsibleError
from datetime import datetime
def get_rrule(self, frequency, kwargs):
    if frequency not in self.frequencies:
        raise AnsibleError('Frequency of {0} is invalid'.format(frequency))
    rrule_kwargs = {'freq': self.frequencies[frequency], 'interval': kwargs.get('every', 1)}
    if 'start_date' in kwargs:
        try:
            rrule_kwargs['dtstart'] = LookupModule.parse_date_time(kwargs['start_date'])
        except Exception as e:
            raise_from(AnsibleError('Parameter start_date must be in the format YYYY-MM-DD [HH:MM:SS]'), e)
    if frequency == 'none':
        rrule_kwargs['count'] = 1
    else:
        if 'end_on' in kwargs:
            end_on = kwargs['end_on']
            if re.match('^\\d+$', end_on):
                rrule_kwargs['count'] = end_on
            else:
                try:
                    rrule_kwargs['until'] = LookupModule.parse_date_time(end_on)
                except Exception as e:
                    raise_from(AnsibleError('Parameter end_on must either be an integer or in the format YYYY-MM-DD [HH:MM:SS]'), e)
        if frequency == 'week' and 'on_days' in kwargs:
            days = []
            for day in kwargs['on_days'].split(','):
                day = day.strip()
                if day not in self.weekdays:
                    raise AnsibleError('Parameter on_days must only contain values {0}'.format(', '.join(self.weekdays.keys())))
                days.append(self.weekdays[day])
            rrule_kwargs['byweekday'] = days
        if frequency == 'month':
            if 'month_day_number' in kwargs and 'on_the' in kwargs:
                raise AnsibleError('Month based frequencies can have month_day_number or on_the but not both')
            if 'month_day_number' in kwargs:
                try:
                    my_month_day = int(kwargs['month_day_number'])
                    if my_month_day < 1 or my_month_day > 31:
                        raise Exception()
                except Exception as e:
                    raise_from(AnsibleError('month_day_number must be between 1 and 31'), e)
                rrule_kwargs['bymonthday'] = my_month_day
            if 'on_the' in kwargs:
                try:
                    occurance, weekday = kwargs['on_the'].split(' ')
                except Exception as e:
                    raise_from(AnsibleError('on_the parameter must be two words separated by a space'), e)
                if weekday not in self.weekdays:
                    raise AnsibleError('Weekday portion of on_the parameter is not valid')
                if occurance not in self.set_positions:
                    raise AnsibleError('The first string of the on_the parameter is not valid')
                rrule_kwargs['byweekday'] = self.weekdays[weekday]
                rrule_kwargs['bysetpos'] = self.set_positions[occurance]
    my_rule = rrule.rrule(**rrule_kwargs)
    timezone = 'America/New_York'
    if 'timezone' in kwargs:
        if kwargs['timezone'] not in pytz.all_timezones:
            raise AnsibleError('Timezone parameter is not valid')
        timezone = kwargs['timezone']
    return_rrule = str(my_rule).replace('\n', ' ').replace('DTSTART:', 'DTSTART;TZID={0}:'.format(timezone))
    if kwargs.get('every', 1) == 1:
        return_rrule = '{0};INTERVAL=1'.format(return_rrule)
    return return_rrule