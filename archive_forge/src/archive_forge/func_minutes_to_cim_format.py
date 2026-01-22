from __future__ import (absolute_import, division, print_function)
import os
from ansible_collections.dellemc.openmanage.plugins.module_utils.dellemc_idrac import iDRACConnection, idrac_auth_params
from ansible.module_utils.basic import AnsibleModule
def minutes_to_cim_format(module, dur_minutes):
    try:
        if dur_minutes < 0:
            module.fail_json(msg='Invalid value for ExposeDuration.')
        MIN_PER_HOUR = 60
        MIN_PER_DAY = 1440
        days = dur_minutes // MIN_PER_DAY
        minutes = dur_minutes % MIN_PER_DAY
        hours = minutes // MIN_PER_HOUR
        minutes = minutes % MIN_PER_HOUR
        if days > 0:
            hours = 23
        cim_format = '{:08d}{:02d}{:02d}00.000000:000'
        cim_time = cim_format.format(days, hours, minutes)
    except Exception:
        module.fail_json(msg='Invalid value for ExposeDuration.')
    return cim_time