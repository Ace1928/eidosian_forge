from __future__ import (absolute_import, division, print_function)
import re
from ansible.errors import AnsibleFilterError
def to_time_unit(human_time, unit='ms', **kwargs):
    """ Return a time unit from a human readable string """
    if human_time == '0':
        return 0
    unit_to_short_form = UNIT_TO_SHORT_FORM
    unit_factors = UNIT_FACTORS
    unit = unit_to_short_form.get(unit.rstrip('s'), unit)
    if unit not in unit_factors:
        raise AnsibleFilterError('to_time_unit() can not convert to the following unit: %s. Available units (singular or plural): %s. Available short units: %s' % (unit, ', '.join(unit_to_short_form.keys()), ', '.join(unit_factors.keys())))
    if 'year' in kwargs:
        unit_factors['y'] = unit_factors['y'][:-1] + [kwargs.pop('year')]
    if 'month' in kwargs:
        unit_factors['mo'] = unit_factors['mo'][:-1] + [kwargs.pop('month')]
    if kwargs:
        raise AnsibleFilterError('to_time_unit() got unknown keyword arguments: %s' % ', '.join(kwargs.keys()))
    result = 0
    for h_time_string in human_time.split():
        res = re.match('(-?\\d+)(\\w+)', h_time_string)
        if not res:
            raise AnsibleFilterError('to_time_unit() can not interpret following string: %s' % human_time)
        h_time_int = int(res.group(1))
        h_time_unit = res.group(2)
        h_time_unit = unit_to_short_form.get(h_time_unit.rstrip('s'), h_time_unit)
        if h_time_unit not in unit_factors:
            raise AnsibleFilterError('to_time_unit() can not interpret following string: %s' % human_time)
        time_in_milliseconds = h_time_int * multiply(unit_factors[h_time_unit])
        result += time_in_milliseconds
    return round(result / multiply(unit_factors[unit]), 12)