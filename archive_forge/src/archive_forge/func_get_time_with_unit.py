from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def get_time_with_unit(time):
    """This method sets seconds in minutes, hours or days."""
    sec_in_min = 60
    sec_in_hour = 60 * 60
    sec_in_day = 24 * 60 * 60
    if time % sec_in_day == 0:
        time = time / sec_in_day
        unit = 'days'
    elif time % sec_in_hour == 0:
        time = time / sec_in_hour
        unit = 'hours'
    else:
        time = time / sec_in_min
        unit = 'minutes'
    return '%s %s' % (time, unit)