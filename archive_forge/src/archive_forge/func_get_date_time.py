from __future__ import absolute_import, division, print_function
import datetime
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.urls import open_url
def get_date_time(start_date, start_time, minutes):
    returned_date = []
    if start_date and start_time:
        try:
            datetime.datetime.strptime(start_date, '%m/%d/%Y')
            returned_date.append(start_date)
        except (NameError, ValueError):
            return (1, None, 'Not a valid start_date format.')
        try:
            datetime.datetime.strptime(start_time, '%H:%M')
            returned_date.append(start_time)
        except (NameError, ValueError):
            return (1, None, 'Not a valid start_time format.')
        try:
            date_time_start = datetime.datetime.strptime(start_time + start_date, '%H:%M%m/%d/%Y')
            delta = date_time_start + datetime.timedelta(minutes=minutes)
            returned_date.append(delta.strftime('%m/%d/%Y'))
            returned_date.append(delta.strftime('%H:%M'))
        except (NameError, ValueError):
            return (1, None, "Couldn't work out a valid date")
    else:
        now = datetime.datetime.utcnow()
        delta = now + datetime.timedelta(minutes=minutes)
        returned_date.append(now.strftime('%m/%d/%Y'))
        returned_date.append(now.strftime('%H:%M'))
        returned_date.append(delta.strftime('%m/%d/%Y'))
        returned_date.append(delta.strftime('%H:%M'))
    return (0, returned_date, None)